from __future__ import annotations

from pathlib import Path
import unicodedata
from typing import Any, Mapping, Literal

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dbfread import DBF


def _strip_accents(text: str) -> str:
	normalized = unicodedata.normalize("NFKD", text)
	return "".join(char for char in normalized if not unicodedata.combining(char))


def clean_text(value: Any) -> str | None:
	if pd.isna(value):
		return None
	text = str(value).strip().lower()
	text = _strip_accents(text)
	text = text.replace("-", " ")
	text = text.replace("'", " ")
	text = " ".join(text.split())
	return text


DEFAULT_CITY_ADM1_MAP: dict[str, str] = {
	"aix en provence": "provence alpes cote d azur",
	"cannes": "provence alpes cote d azur",
	"metz": "lorraine",
	"toulon": "provence alpes cote d azur",
	"nice": "provence alpes cote d azur",
	"lyon": "rhone alpes",
	"nimes": "languedoc roussillon",
	"toulouse": "midi pyrenees",
	"villeurbanne": "rhone alpes",
	"ajaccio": "corse",
	"brest": "bretagne",
	"caen": "basse normandie",
	"reims": "champagne ardenne",
	"rennes": "bretagne",
	"angers": "pays de la loire",
	"limoges": "limousin",
	"le mans": "pays de la loire",
	"paris": "ile de france",
	"la rochelle": "poitou charentes",
	"orleans": "centre",
	"grenoble": "rhone alpes",
	"perpignan": "languedoc roussillon",
	"bordeaux": "aquitaine",
	"strasbourg": "alsace",
	"lille": "nord pas de calais",
	"saint etienne": "rhone alpes",
	"nantes": "pays de la loire",
	"clermont ferrand": "auvergne",
	"dijon": "bourgogne",
	"nancy": "lorraine",
	"saint malo": "bretagne",
	"amiens": "picardie",
	"le havre": "haute normandie",
	"tours": "centre",
	"annecy": "rhone alpes",
	"marseille": "provence alpes cote d azur",
	"montpellier": "languedoc roussillon",
}


DEFAULT_MANUAL_MISSING: list[dict[str, Any]] = [
	{
		"NAME": "Lyon",
		"name_clean": "lyon",
		"ADM1": "RHONE-ALPES",
		"ADM2": "RHONE",
		"LAT": 45.7578,
		"LONG": 4.8320,
	},
	{
		"NAME": "Nîmes",
		"name_clean": "nimes",
		"ADM1": "LANGUEDOC-ROUSSILLON",
		"ADM2": "GARD",
		"LAT": 43.8367,
		"LONG": 4.3601,
	},
]


def load_city_gazetteer_from_dbf(dbf_path: str | Path, class_filter: str = "P") -> pd.DataFrame:
	dbf_path = Path(dbf_path)
	cities = pd.DataFrame(iter(DBF(dbf_path, load=True)))
	if class_filter:
		cities = cities.loc[cities["F_CLASS"] == class_filter].copy()

	cities["name_clean"] = cities["NAME"].apply(clean_text)
	cities["adm1_clean"] = cities["ADM1"].apply(clean_text)
	cities["adm2_clean"] = cities["ADM2"].apply(clean_text)
	return cities


def pick_best_city_row(
	city_clean: str,
	cities_df: pd.DataFrame,
	city_adm1_map: Mapping[str, str] | None = None,
) -> pd.Series | None:
	subset = cities_df.loc[cities_df["name_clean"] == city_clean].copy()
	if subset.empty:
		return None

	target_adm1 = (city_adm1_map or {}).get(city_clean)
	if target_adm1 is not None:
		subset_adm1 = subset.loc[subset["adm1_clean"] == target_adm1].copy()
		if not subset_adm1.empty:
			return subset_adm1.iloc[0]

	if len(subset) == 1:
		return subset.iloc[0]
	return subset.iloc[0]


def build_city_reference(
	claims_df: pd.DataFrame,
	cities_df: pd.DataFrame,
	*,
	location_clean_col: str = "location_clean",
	city_adm1_map: Mapping[str, str] | None = None,
	manual_missing: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
	selected_rows: list[pd.Series] = []
	unique_locations = sorted(claims_df[location_clean_col].dropna().unique())
	for city_clean in unique_locations:
		row = pick_best_city_row(city_clean=str(city_clean), cities_df=cities_df, city_adm1_map=city_adm1_map)
		if row is not None:
			selected_rows.append(row)

	city_reference = pd.DataFrame(selected_rows).copy()

	if manual_missing:
		manual_df = pd.DataFrame(manual_missing)
		if "name_clean" not in manual_df.columns and "NAME" in manual_df.columns:
			manual_df["name_clean"] = manual_df["NAME"].apply(clean_text)
		if "adm1_clean" not in manual_df.columns and "ADM1" in manual_df.columns:
			manual_df["adm1_clean"] = manual_df["ADM1"].apply(clean_text)
		if "adm2_clean" not in manual_df.columns and "ADM2" in manual_df.columns:
			manual_df["adm2_clean"] = manual_df["ADM2"].apply(clean_text)
		city_reference = pd.concat([city_reference, manual_df], ignore_index=True)

	if not city_reference.empty:
		city_reference = city_reference.drop_duplicates(subset=["name_clean"], keep="last").copy()

	return city_reference


def merge_claims_with_city_reference(
	claims_df: pd.DataFrame,
	city_reference: pd.DataFrame,
	*,
	location_clean_col: str = "location_clean",
) -> pd.DataFrame:
	ref_cols = ["NAME", "ADM1", "ADM2", "LAT", "LONG", "name_clean", "adm1_clean", "adm2_clean"]
	available_cols = [col for col in ref_cols if col in city_reference.columns]
	return claims_df.merge(
		city_reference.loc[:, available_cols],
		left_on=location_clean_col,
		right_on="name_clean",
		how="left",
	)


def prepare_claims_location_mapping(
	claims_df: pd.DataFrame,
	*,
	dbf_path: str | Path,
	location_col: str = "location",
	location_clean_col: str = "location_clean",
	city_adm1_map: Mapping[str, str] | None = DEFAULT_CITY_ADM1_MAP,
	manual_missing: list[dict[str, Any]] | None = DEFAULT_MANUAL_MISSING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	claims_map = claims_df.copy()
	claims_map[location_clean_col] = claims_map[location_col].apply(clean_text)

	cities = load_city_gazetteer_from_dbf(dbf_path)
	city_reference = build_city_reference(
		claims_map,
		cities,
		location_clean_col=location_clean_col,
		city_adm1_map=city_adm1_map,
		manual_missing=manual_missing,
	)
	map_df = merge_claims_with_city_reference(claims_map, city_reference, location_clean_col=location_clean_col)
	return map_df, city_reference


# =========================================================
# Mapping helpers
# =========================================================

MapLevel = Literal["city", "dep", "region"]


def get_fraud_color_labels() -> list[str]:
	return [
		# "#084081",  # 0%
		# "#0868ac",  # 0-10%
		# "#2b8cbe",  # 10-20%
		# "#4eb3d3",  # 20-30%
		# "#a8ddb5",  # 30-40%
		# "#ccebc5",  # 40-50%
		# "#f7fcb9",  # 50-60%
		# "#fed976",  # 60-70%
		# "#fd8d3c",  # 70-80%
		# "#e31a1c",  # 80-100%
		"#084081",
		"#0868ac",
		"#2b8cbe",
		"#4eb3d3",
		"#7bccc4",
		"#a8ddb5",
		"#ccebc5",
		"#f7fcb9",
		"#fed976",
		"#fd8d3c",
	]


def get_fraud_bins() -> list[float]:
	return [0.05, 0.055, 0.06, 0.065, 0.07, 0.0725, 0.075, 0.08, 0.085, 0.09, 0.095]


def aggregate_fraud_for_map(
	map_df: pd.DataFrame,
	*,
	level: MapLevel,
	fraud_col: str = "is_fraud",
	location_col: str = "location",
	lat_col: str = "LAT",
	lon_col: str = "LONG",
	adm1_col: str = "ADM1",
	adm2_col: str = "ADM2",
) -> pd.DataFrame:
	if level == "city":
		agg = (
			map_df.dropna(subset=[lat_col, lon_col, location_col])
			.groupby([location_col, lat_col, lon_col], as_index=False)
			.agg(
				n_claims=(fraud_col, "size"),
				fraud_rate=(fraud_col, "mean"),
				fraud_count=(fraud_col, "sum"),
			)
		)
		agg["color"] = pd.cut(
			agg["fraud_rate"],
			bins=get_fraud_bins(),
			labels=get_fraud_color_labels(),
			include_lowest=True,
		)
		return agg

	if level == "dep":
		agg = (
			map_df.dropna(subset=[adm2_col])
			.groupby(adm2_col, as_index=False)
			.agg(
				n_claims=(fraud_col, "size"),
				fraud_rate=(fraud_col, "mean"),
				fraud_count=(fraud_col, "sum"),
			)
		)
		agg["zone_name"] = agg[adm2_col]
		agg["zone_clean"] = agg[adm2_col].apply(clean_text)
		agg["color"] = pd.cut(
			agg["fraud_rate"],
			bins=get_fraud_bins(),
			labels=get_fraud_color_labels(),
			include_lowest=True,
		)
		return agg

	if level == "region":
		agg = (
			map_df.dropna(subset=[adm1_col])
			.groupby(adm1_col, as_index=False)
			.agg(
				n_claims=(fraud_col, "size"),
				fraud_rate=(fraud_col, "mean"),
				fraud_count=(fraud_col, "sum"),
			)
		)
		agg["zone_name"] = agg[adm1_col]
		agg["zone_clean"] = agg[adm1_col].apply(clean_text)
		agg["color"] = pd.cut(
			agg["fraud_rate"],
			bins=get_fraud_bins(),
			labels=get_fraud_color_labels(),
			include_lowest=True,
		)
		return agg

	raise ValueError("level must be one of: 'city', 'dep', 'region'")


def load_admin_polygons(
	geo_path: str | Path,
	*,
	name_col: str = "nom",
) -> gpd.GeoDataFrame:
	geo_path = Path(geo_path)
	gdf = gpd.read_file(geo_path)
	gdf["zone_clean"] = gdf[name_col].apply(clean_text)
	return gdf


def build_map_geodf(
    map_df: pd.DataFrame,
    *,
    missing_color: str = "#E4E1E1",
    level: MapLevel,
    geo_path: str | Path | None = None,
    geo_name_col: str = "nom",
) -> gpd.GeoDataFrame:
    agg = aggregate_fraud_for_map(map_df, level=level)

    if level == "city":
        gdf = gpd.GeoDataFrame(
            agg,
            geometry=gpd.points_from_xy(agg["LONG"], agg["LAT"]),
            crs="EPSG:4326",
        )
        return gdf

    if geo_path is None:
        raise ValueError("geo_path is required for 'dep' and 'region' levels")

    polygons = load_admin_polygons(geo_path, name_col=geo_name_col)
    gdf = polygons.merge(
        agg[["zone_name", "zone_clean", "fraud_rate", "n_claims", "fraud_count", "color"]],
        on="zone_clean",
        how="left",
    )
 
	# fill missing colors with a default value (e.g., gray)
    if pd.api.types.is_categorical_dtype(gdf["color"]):
        gdf["color"] = gdf["color"].cat.add_categories([missing_color]).fillna(missing_color)
    else:
        gdf["color"] = gdf["color"].fillna(missing_color)
    
    return gdf



def plot_fraud_map(
	map_df: pd.DataFrame,
	*,
	level: MapLevel,
	geo_path: str | Path | None = None,
	geo_name_col: str = "nom",
	figsize: tuple[int, int] = (11, 11),
	title: str | None = None,
	markersize_scale: float = 4.0,
	missing_color: str = "#E4E1E1",
	edgecolor: str = "black",
	linewidth: float = 0.5,
	legend: bool = True,
	ax: plt.Axes | None = None,
) -> plt.Axes:
	gdf = build_map_geodf(
		map_df,
		level=level,
		geo_path=geo_path,
		geo_name_col=geo_name_col,
	)

	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)

	plot_colors = gdf["color"].astype(object).where(gdf["color"].notna(), missing_color)
    
	if level == "city":
		gdf.plot(
			ax=ax,
			color=plot_colors,
			markersize=gdf["n_claims"].fillna(1) * markersize_scale,
			alpha=0.75,
		)
	else:
		gdf.plot(
			ax=ax,
			color=plot_colors,
			edgecolor=edgecolor,
			linewidth=linewidth,
		)

	if legend:
		legend_elements = [
			Patch(facecolor="#e31a1c", edgecolor="black", label="80-100%"),
			Patch(facecolor="#fd8d3c", edgecolor="black", label="70-80%"),
			Patch(facecolor="#fed976", edgecolor="black", label="60-70%"),
			Patch(facecolor="#f7fcb9", edgecolor="black", label="50-60%"),
			Patch(facecolor="#ccebc5", edgecolor="black", label="40-50%"),
			Patch(facecolor="#a8ddb5", edgecolor="black", label="30-40%"),
			Patch(facecolor="#4eb3d3", edgecolor="black", label="20-30%"),
			Patch(facecolor="#2b8cbe", edgecolor="black", label="10-20%"),
			Patch(facecolor="#0868ac", edgecolor="black", label="0-10%"),
			Patch(facecolor="#084081", edgecolor="black", label="0%"),
			Patch(facecolor=missing_color, edgecolor="black", label="No data"),
		]
		ax.legend(handles=legend_elements, title="Fraud rate", loc="lower left")

	if title is None:
		title = {
			"city": "Fraud rate by city",
			"dep": "Fraud rate by department",
			"region": "Fraud rate by region",
		}[level]

	ax.set_title(title)
	ax.set_axis_off()
	return ax