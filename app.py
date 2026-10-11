from pathlib import Path
from io import BytesIO
import base64
import math

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from folium.raster_layers import ImageOverlay
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium


# =========================================================
# KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="LULC Jawa Timur | Random Forest",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
GEOJSON_PATH = DATA_DIR / "jawatimur.geojson"

# Nama utama diutamakan. Jika berbeda, aplikasi mencari raster .tif di folder output.
EXPECTED_RASTER = OUTPUT_DIR / "LULC_Jawa_Timur_RandomForest.tif"


CLASS_INFO = {
    1: {
        "name": "Build up",
        "label_id": "Kawasan terbangun",
        "color": "#808080",
        "description": "Area yang didominasi bangunan, permukiman, jalan, atau permukaan buatan.",
    },
    2: {
        "name": "Farm land",
        "label_id": "Lahan pertanian",
        "color": "#FFD700",
        "description": "Area yang diberi label lahan pertanian pada polygon sampel; tidak otomatis berarti seluruh piksel adalah sawah.",
    },
    3: {
        "name": "Forest",
        "label_id": "Hutan",
        "color": "#228B22",
        "description": "Area bervegetasi yang diberi label hutan pada data sampel.",
    },
    4: {
        "name": "Mangrove",
        "label_id": "Hutan mangrove",
        "color": "#00CED1",
        "description": "Vegetasi mangrove di kawasan pesisir sesuai polygon sampel.",
    },
    5: {
        "name": "Water body",
        "label_id": "Badan air",
        "color": "#2196F3",
        "description": "Piksel yang diprediksi sebagai badan air. Tanpa mask wilayah yang tepat, kelas ini dapat mencakup perairan laut.",
    },
}
VALID_NAMES = {v["name"].casefold() for v in CLASS_INFO.values()}

CLASS_ALIASES = {
    "fram land": "farm land",
    "farmland": "farm land",
    "farm land": "farm land",
    "built-up": "build up",
    "built up": "build up",
    "builtup": "build up",
    "kawasan terbangun": "build up",
    "lahan pertanian": "farm land",
    "hutan": "forest",
    "hutan mangrove": "mangrove",
    "badan air": "water body",
    "waterbody": "water body",
    "water body": "water body",
    "none": "unknown",
    "nan": "unknown",
    "null": "unknown",
    "": "unknown",
}


# =========================================================
# UTILITAS FILE
# =========================================================
def find_raster():
    if EXPECTED_RASTER.exists():
        return EXPECTED_RASTER
    if OUTPUT_DIR.exists():
        candidates = sorted(
            list(OUTPUT_DIR.glob("*.tif")) + list(OUTPUT_DIR.glob("*.tiff")),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def find_true_color():
    if not DATA_DIR.exists():
        return None
    patterns = ("*True_color*.tif", "*True_color*.tiff", "*True*color*.tif", "*True*color*.tiff")
    candidates = []
    for pattern in patterns:
        candidates.extend(DATA_DIR.glob(pattern))
    candidates = list(dict.fromkeys(candidates))
    return candidates[0] if candidates else None


RASTER_PATH = find_raster()
TRUE_COLOR_PATH = find_true_color()


# =========================================================
# HEADER DAN GAYA
# =========================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1600px;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(115deg, #0b2538 0%, #14534c 58%, #23816d 100%);
        color: white; border: 1px solid rgba(255,255,255,.12);
    }
    .hero h1 {font-size: 2.1rem; margin: 0 0 .35rem 0; color: white;}
    .hero p {font-size: 1rem; margin: .2rem 0 0 0; color: #e0f2ed;}
    .section-note {color: #aab5c2; font-size: .92rem;}
    .legend-row {display:flex; align-items:flex-start; gap:12px; padding:10px 0;
        border-bottom:1px solid rgba(128,128,128,.2);}
    .swatch {width:18px; height:18px; border-radius:5px; flex:0 0 18px; margin-top:3px;
        border:1px solid rgba(255,255,255,.35);}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <h1>🛰️ Dashboard Tutupan Lahan Jawa Timur</h1>
      <p>Sentinel-2A Level-2A · Supervised Classification · Random Forest</p>
      <p style="margin-top:.6rem;font-size:.92rem">
        Visualisasi raster klasifikasi, citra True Color, polygon sampel, statistik kelas,
        dan metadata berkas proyek.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# VALIDASI BERKAS
# =========================================================
missing = []
if RASTER_PATH is None:
    missing.append(f"Raster klasifikasi tidak ditemukan di {OUTPUT_DIR}")
if not GEOJSON_PATH.exists():
    missing.append(f"GeoJSON tidak ditemukan: {GEOJSON_PATH}")
if TRUE_COLOR_PATH is None:
    missing.append(f"Citra Sentinel True Color tidak ditemukan di {DATA_DIR}")

if missing:
    st.error("Ada berkas proyek yang belum ditemukan:")
    for item in missing:
        st.write(f"- {item}")
    st.info(
        "Pastikan struktur folder: app.py, data/jawatimur.geojson, "
        "data/*True_color*.tiff, dan output/*.tif."
    )
    st.stop()


def rgba_data_url(rgba):
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def bounds_wgs84(src):
    if src.crs is None:
        raise ValueError(f"Raster tidak memiliki CRS: {src.name}")
    west, south, east, north = transform_bounds(
        src.crs,
        "EPSG:4326",
        src.bounds.left,
        src.bounds.bottom,
        src.bounds.right,
        src.bounds.top,
        densify_pts=21,
    )
    return [[south, west], [north, east]]


@st.cache_data(show_spinner="Membaca raster klasifikasi...")
def load_classification(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Raster klasifikasi tidak memiliki CRS.")
        scale = min(1.0, 1500 / max(src.width, src.height))
        width = max(1, round(src.width * scale))
        height = max(1, round(src.height * scale))
        small = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        small_mask = src.dataset_mask(out_shape=(height, width)) > 0
        full = src.read(1)
        full_mask = src.dataset_mask() > 0
        bounds = bounds_wgs84(src)
        meta = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": bounds,
            "pixel_size": src.res,
            "dtype": str(src.dtypes[0]),
            "count": src.count,
        }

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for code, info in CLASS_INFO.items():
        color = info["color"].lstrip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        selected = (small == code) & small_mask
        rgba[selected, :3] = rgb
        rgba[selected, 3] = 220
    return rgba_data_url(rgba), bounds, full, full_mask, meta


@st.cache_data(show_spinner="Membaca citra Sentinel-2A...")
def load_true_color(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Citra True Color tidak memiliki CRS.")
        if src.count < 3:
            raise ValueError("Citra True Color harus memiliki minimal 3 band (RGB).")
        scale = min(1.0, 1500 / max(src.width, src.height))
        width = max(1, round(src.width * scale))
        height = max(1, round(src.height * scale))
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        mask = src.dataset_mask(out_shape=(height, width)) > 0
        bounds = bounds_wgs84(src)
        meta = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "dtype": str(src.dtypes[0]),
            "count": src.count,
            "pixel_size": src.res,
        }

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for band_idx in range(3):
        band = rgb[band_idx]
        values = band[np.isfinite(band) & mask]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            if high > low:
                rgba[:, :, band_idx] = (
                    np.clip((band - low) / (high - low), 0, 1) * 255
                ).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return rgba_data_url(rgba), bounds, meta


@st.cache_data(show_spinner="Membaca polygon sampel GeoJSON...")
def load_samples(path):
    data = gpd.read_file(path)
    if data.crs is None:
        raise ValueError("GeoJSON tidak memiliki CRS.")
    data = data.to_crs("EPSG:4326")
    source_col = next(
        (c for c in ["class", "nama", "kelas", "class_name", "name", "label"] if c in data.columns),
        None,
    )
    if source_col is None:
        raise ValueError(
            "Kolom label kelas tidak ditemukan. Kolom yang tersedia: "
            + ", ".join(map(str, data.columns))
        )

    # Tangani nilai null sebelum mengubahnya menjadi string, agar tidak dianggap label 'None'.
    raw = data[source_col]
    data["class_original"] = raw.where(raw.notna(), "(label kosong)").astype(str).str.strip()
    normalized = raw.fillna("").astype(str).str.strip().str.casefold()
    normalized = normalized.replace(CLASS_ALIASES)
    data["class"] = normalized
    data["source_label_column"] = source_col
    return data


try:
    class_image, class_bounds, raster_data, raster_valid, raster_meta = load_classification(str(RASTER_PATH))
    true_image, true_bounds, true_meta = load_true_color(str(TRUE_COLOR_PATH))
    samples_all = load_samples(str(GEOJSON_PATH))
except Exception as exc:
    st.error(f"Gagal membaca data: {exc}")
    st.stop()


samples = samples_all[samples_all["class"].isin(VALID_NAMES)].copy()
excluded_samples = samples_all[~samples_all.index.isin(samples.index)].copy()
valid_values = raster_data[raster_valid]
total_valid = int(raster_valid.sum())
total_pixels = int(raster_data.size)
nodata_pixels = total_pixels - total_valid
class_counts = {code: int(np.count_nonzero(valid_values == code)) for code in CLASS_INFO}
unknown_values = sorted(int(v) for v in np.unique(valid_values) if v not in CLASS_INFO)


# =========================================================
# RINGKASAN UTAMA
# =========================================================
st.subheader("Ringkasan data")
st.caption("Angka dihitung dari berkas raster dan GeoJSON yang sedang dibaca aplikasi.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Polygon sampel valid", f"{len(samples):,}")
c2.metric("Kelas sampel valid", f"{samples['class'].nunique()}")
c3.metric("Piksel valid raster", f"{total_valid:,}")
c4.metric("Piksel NoData", f"{nodata_pixels:,}")

if len(samples) and samples["class"].nunique() != len(CLASS_INFO):
    st.warning(
        f"GeoJSON valid berisi {samples['class'].nunique()} dari {len(CLASS_INFO)} kelas model. "
        "Periksa nama label kelas pada GeoJSON."
    )
if unknown_values:
    st.warning(
        f"Raster memiliki kode kelas yang belum dipetakan: {unknown_values}. "
        "Cocokkan kode raster dengan kelas di notebook."
    )
if len(excluded_samples):
    with st.expander(f"Label sampel yang belum masuk hitungan kelas ({len(excluded_samples)} fitur)"):
        st.write(
            "Fitur berikut tidak cocok dengan lima label kelas model. "
            "Periksa nilai label pada GeoJSON; label kosong tidak otomatis dimasukkan ke kelas mana pun."
        )
        st.dataframe(
            excluded_samples["class_original"].value_counts()
            .rename_axis("Label asli")
            .reset_index(name="Jumlah"),
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# KOMPOSISI KELAS
# =========================================================
st.subheader("Komposisi hasil klasifikasi")
st.markdown(
    '<p class="section-note">Persentase dihitung dari piksel raster valid. '
    'Ini merupakan distribusi prediksi model, bukan otomatis persentase resmi luas provinsi.</p>',
    unsafe_allow_html=True,
)

stats_rows = []
for code, info in CLASS_INFO.items():
    count = class_counts[code]
    stats_rows.append({
        "Kode": code,
        "Kelas": info["name"],
        "Jumlah piksel": count,
        "Persentase (%)": round(count / total_valid * 100, 2) if total_valid else 0,
        "Warna": info["color"],
    })
stats_df = pd.DataFrame(stats_rows)
col_table, col_bar = st.columns([1.15, 1])
with col_table:
    st.dataframe(
        stats_df[["Kode", "Kelas", "Jumlah piksel", "Persentase (%)"]],
        hide_index=True,
        use_container_width=True,
    )
with col_bar:
    chart_df = stats_df.set_index("Kelas")[["Persentase (%)"]]
    st.bar_chart(chart_df, horizontal=True, height=280)

with st.expander("Apa arti statistik ini?"):
    st.markdown(
        """
        - **Jumlah piksel:** jumlah piksel raster yang memiliki kode kelas tersebut.
        - **Persentase:** jumlah piksel kelas dibagi seluruh piksel valid, dikali 100%.
        - **Piksel valid:** piksel yang tidak ditandai NoData oleh raster.
        - **NoData:** piksel yang tidak memiliki data valid untuk dihitung.
        - **Polygon sampel:** geometri yang diberi label untuk sampel pelatihan; bukan otomatis batas administrasi provinsi.
        """
    )


# =========================================================
# LEGENDA KELAS
# =========================================================
st.subheader("Legenda dan definisi kelas")
legend_cols = st.columns(5)
for col, (code, info) in zip(legend_cols, CLASS_INFO.items()):
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,.28);border-radius:12px;padding:14px;min-height:190px">
              <div style="width:26px;height:26px;background:{info['color']};border-radius:7px;margin-bottom:10px;border:1px solid rgba(128,128,128,.35)"></div>
              <div style="font-weight:700;font-size:1rem">{info['name']}</div>
              <div style="font-size:.86rem;opacity:.8;margin:.15rem 0 .55rem">{info['label_id']}</div>
              <div style="font-size:.82rem;line-height:1.35">{info['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PETA INTERAKTIF
# =========================================================
st.subheader("Peta interaktif")
st.caption(
    "Gunakan kontrol layer di kanan atas peta untuk menampilkan atau menyembunyikan citra, "
    "hasil klasifikasi, dan polygon sampel."
)
with st.sidebar:
    st.header("Kontrol peta")
    show_true_color = st.checkbox("Citra Sentinel-2A True Color", value=True)
    show_classification = st.checkbox("Prediksi Random Forest", value=True)
    show_samples = st.checkbox("Polygon sampel GeoJSON", value=True)
    selected_names = st.multiselect(
        "Kelas polygon sampel",
        [info["name"] for info in CLASS_INFO.values()],
        default=[info["name"] for info in CLASS_INFO.values()],
    )
    opacity = st.slider("Transparansi hasil klasifikasi", 0.2, 1.0, 0.82, 0.05)
    st.divider()
    st.caption("Sumber: raster klasifikasi, citra True Color, dan GeoJSON di folder proyek.")

south, west = class_bounds[0]
north, east = class_bounds[1]
center = [(south + north) / 2, (west + east) / 2]
map_view = folium.Map(
    location=center,
    zoom_start=7,
    tiles="OpenStreetMap",
    control_scale=True,
    prefer_canvas=True,
)

if show_true_color:
    ImageOverlay(
        true_image,
        bounds=true_bounds,
        name="Sentinel-2A True Color",
        opacity=1,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(map_view)

if show_classification:
    ImageOverlay(
        class_image,
        bounds=class_bounds,
        name="Prediksi Random Forest",
        opacity=opacity,
        interactive=False,
        cross_origin=False,
        zindex=2,
    ).add_to(map_view)

if show_samples:
    for _, info in CLASS_INFO.items():
        if info["name"] not in selected_names:
            continue
        subset = samples[samples["class"] == info["name"].casefold()]
        if subset.empty:
            continue
        feature_group = folium.FeatureGroup(name=f"Sampel: {info['name']}", show=True)
        folium.GeoJson(
            subset.to_json(),
            name=info["name"],
            style_function=lambda feature, color=info["color"]: {
                "color": color,
                "weight": 2,
                "fillColor": color,
                "fillOpacity": 0.24,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["class_original"],
                aliases=["Label sampel:"],
            ),
        ).add_to(feature_group)
        feature_group.add_to(map_view)

folium.LayerControl(collapsed=False).add_to(map_view)
map_view.fit_bounds([[south, west], [north, east]])
st_folium(map_view, height=650, use_container_width=True, key="lulc_map_v3")


# =========================================================
# METADATA DAN CATATAN METODE
# =========================================================
st.subheader("Metode, metadata, dan batasan")
with st.expander("Metode yang digunakan", expanded=True):
    st.markdown(
        """
        **Supervised Classification dengan algoritma Random Forest**

        1. Polygon GeoJSON menyediakan label kelas sampel.
        2. Nilai spektral dari band Sentinel-2A digunakan sebagai fitur masukan model.
        3. Random Forest mempelajari hubungan antara fitur spektral dan label sampel.
        4. Model yang sudah dilatih menghasilkan prediksi kelas per piksel.
        5. Raster prediksi ditampilkan pada peta dan diringkas menjadi jumlah piksel serta persentase kelas.

        Dashboard ini menampilkan raster hasil klasifikasi yang sudah ada; aplikasi ini tidak melatih ulang model.
        Akurasi tidak dapat disimpulkan dari peta saja dan perlu dinilai menggunakan data uji serta metrik evaluasi dari notebook.
        """
    )

with st.expander("Metadata raster klasifikasi"):
    st.write(f"**Nama berkas:** `{RASTER_PATH.name}`")
    st.write(f"**CRS:** `{raster_meta['crs']}`")
    st.write(f"**Dimensi:** {raster_meta['width']:,} × {raster_meta['height']:,} piksel")
    st.write(f"**Jumlah band:** {raster_meta['count']}")
    st.write(f"**Tipe data:** `{raster_meta['dtype']}`")
    st.write(f"**Nilai NoData metadata:** `{raster_meta['nodata']}`")
    st.write(f"**Resolusi/grid:** `{raster_meta['pixel_size']}`")
    st.write(f"**Batas raster WGS84:** `{raster_meta['bounds']}`")

with st.expander("Metadata citra Sentinel-2A True Color"):
    st.write(f"**Nama berkas:** `{TRUE_COLOR_PATH.name}`")
    st.write(f"**CRS:** `{true_meta['crs']}`")
    st.write(f"**Dimensi:** {true_meta['width']:,} × {true_meta['height']:,} piksel")
    st.write(f"**Jumlah band:** {true_meta['count']}")
    st.write(f"**Tipe data:** `{true_meta['dtype']}`")
    st.write(f"**Resolusi/grid:** `{true_meta['pixel_size']}`")

with st.expander("Informasi GeoJSON sampel"):
    st.write(f"**Nama berkas:** `{GEOJSON_PATH.name}`")
    st.write(f"**Kolom label yang digunakan:** `{samples_all['source_label_column'].iloc[0]}`")
    st.write(f"**Jumlah seluruh fitur:** {len(samples_all):,}")
    st.write(f"**Fitur berlabel valid:** {len(samples):,}")
    st.write(f"**Fitur yang perlu diperiksa:** {len(excluded_samples):,}")
    st.write("**Jumlah fitur per label valid:**")
    if len(samples):
        st.dataframe(
            samples["class_original"].value_counts().rename_axis("Label").reset_index(name="Jumlah"),
            hide_index=True,
            use_container_width=True,
        )

st.warning(
    "Batasan interpretasi: polygon GeoJSON yang digunakan adalah polygon sampel, bukan batas administrasi "
    "Provinsi Jawa Timur. Untuk menghitung luas resmi per provinsi atau memastikan hasil hanya mencakup daratan, "
    "gunakan batas administrasi/mask yang benar dan verifikasi kesesuaian spasialnya. "
    "Distribusi piksel prediksi tidak menggantikan evaluasi akurasi model."
)

st.caption("Dashboard eksplorasi LULC · Sentinel-2A · Random Forest")
