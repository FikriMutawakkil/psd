from pathlib import Path
import json, base64
from io import BytesIO
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import streamlit as st
import folium
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
RASTER_OPTIONS = [APP_DIR/'data/lulc_jawa_timur_rf_revisi.tif', APP_DIR/'data/sentinel_bands/lulc_jawa_timur_rf_revisi.tif', APP_DIR/'lulc_jawa_timur_rf_revisi.tif']
GEOJSON_OPTIONS = [APP_DIR/'data/geojson/jawatimur.geojson', APP_DIR/'data/jawatimur.geojson', APP_DIR/'jawatimur.geojson']
def first_existing(paths): return next((p for p in paths if p.exists()), None)

CLASSES = {
  0: {'name':'Farm Land','label':'farm land','color':'#F4C542'},
  1: {'name':'Build Up','label':'build up','color':'#F06449'},
  2: {'name':'Water Body','label':'water body','color':'#3198D4'},
  3: {'name':'Forest','label':'forest','color':'#35A66F'},
  4: {'name':'Hutan Mangrove','label':'hutan mangrove','color':'#20B8AE'},
}
# Validation metrics from the GEE Console results shared in the conversation.
CM = np.array([[18,1,0,3,0],[5,13,0,0,0],[1,0,12,1,1],[1,0,0,17,0],[0,0,1,0,9]], dtype=int)
OA, KAPPA = 0.8313253012048193, 0.7853712596970817

st.set_page_config(page_title='GeoVision | LULC Jawa Timur', page_icon='🌿', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:linear-gradient(180deg,#F1FAF6 0%,#F7FAFF 55%,#FFFFFF 100%)}
.block-container{max-width:1500px;padding-top:1.3rem}.hero{background:linear-gradient(120deg,#0D5C4B,#168A70 55%,#64CDA9);border-radius:24px;padding:28px 32px;color:white;margin-bottom:20px;box-shadow:0 12px 30px #0d5c4b20}
.hero h1{font-family:Manrope,sans-serif;color:white;font-size:2.15rem;margin:0 0 8px}.hero p{color:#E8FFF5;margin:0}.eyebrow{color:#BFF5DE;font-size:.76rem;letter-spacing:.13em;font-weight:700;margin-bottom:8px}
.metric{background:white;border:1px solid #DDEDE5;border-radius:17px;padding:16px 18px;box-shadow:0 5px 18px #1a4b3c0c;min-height:108px}.mlabel{color:#668078;font-size:.82rem;font-weight:600}.mvalue{color:#164C3E;font-family:Manrope,sans-serif;font-size:1.65rem;font-weight:800;margin-top:7px}.mnote{color:#7C928B;font-size:.75rem;margin-top:3px}
.section{font-family:Manrope,sans-serif;color:#174C40;font-size:1.25rem;font-weight:800;margin:1rem 0 .65rem}.note{background:#EFF8F4;border:1px solid #D5EDE2;border-radius:14px;padding:13px 15px;color:#28594D}[data-testid="stSidebar"]{background:#F0F8F4;border-right:1px solid #DFEEE7}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">GEOSPATIAL INTELLIGENCE · SENTINEL-2</div><h1>🌿 GeoVision — LULC Jawa Timur</h1><p>Dashboard interaktif tutupan lahan: kontrol lima kelas, transparansi, sampel GeoJSON, dan evaluasi Random Forest.</p></div>''', unsafe_allow_html=True)

raster_path, geo_path = first_existing(RASTER_OPTIONS), first_existing(GEOJSON_OPTIONS)
with st.sidebar:
    st.markdown('## 🎛️ Kontrol Peta')
    basemap = st.radio('Peta dasar', ['Citra Satelit','OpenStreetMap'])
    st.markdown('### 🗺️ Kelas raster')
    enabled = {i: st.checkbox(v['name'], value=True, key=f'cls_{i}') for i,v in CLASSES.items()}
    opacity = st.slider('Transparansi raster', 0, 100, 72, 5)
    show_samples = st.checkbox('Tampilkan sampel GeoJSON', True)
    sample_opacity = st.slider('Transparansi sampel', 0, 100, 25, 5)
    st.markdown('---'); st.caption('Legenda')
    for v in CLASSES.values(): st.markdown(f'<span style="color:{v["color"]};font-weight:700">● {v["name"]}</span>', unsafe_allow_html=True)

if raster_path is None:
    st.error('Raster tidak ditemukan. Letakkan lulc_jawa_timur_rf_revisi.tif di folder data/ atau data/sentinel_bands/.'); st.stop()

@st.cache_data(show_spinner='Membaca raster dan menyiapkan preview…')
def read_raster(path, max_side=1300):
    with rasterio.open(path) as src:
        factor=min(1, max_side/max(src.width,src.height)); h=max(1,int(src.height*factor)); w=max(1,int(src.width*factor))
        arr=src.read(1,out_shape=(h,w),resampling=Resampling.nearest); b=src.bounds
        bounds=[b.bottom,b.left,b.top,b.right]; nodata=src.nodata
        counts={i:0 for i in CLASSES}
        for _,win in src.block_windows(1):
            block=src.read(1,window=win); vals,nums=np.unique(block,return_counts=True)
            for val,num in zip(vals,nums):
                if int(val) in counts: counts[int(val)]+=int(num)
        return arr,bounds,nodata,counts,src.crs.to_string(),src.width,src.height
try: raster,bounds,nodata,counts,crs,original_w,original_h=read_raster(str(raster_path))
except Exception as e: st.error(f'Raster gagal dibaca: {e}'); st.stop()
valid_count=sum(counts.values()); names=[CLASSES[i]['name'] for i in range(5)]
m1,m2,m3,m4=st.columns(4)
def metric(col,label,value,note): col.markdown(f'<div class="metric"><div class="mlabel">{label}</div><div class="mvalue">{value}</div><div class="mnote">{note}</div></div>',unsafe_allow_html=True)
metric(m1,'Overall Accuracy',f'{OA*100:.2f}%','Hasil validasi GEE'); metric(m2,'Cohen’s Kappa',f'{KAPPA:.4f}','Kesepakatan di luar peluang'); metric(m3,'Sampel validasi',str(int(CM.sum())),'Sampel pada confusion matrix'); metric(m4,'Kelas LULC','5','Kelas tutupan lahan')

st.markdown('<div class="section">🗺️ Peta klasifikasi interaktif</div>',unsafe_allow_html=True)
st.caption('Aktif/nonaktif kelas di sidebar dan geser transparansi untuk melihat citra satelit di balik raster.')
south,west,north,east=bounds; center=[(south+north)/2,(west+east)/2]
m=folium.Map(location=center,zoom_start=7,tiles=None,control_scale=True,prefer_canvas=True)
satellite={'tiles':'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}','attr':'Tiles © Esri','name':'Citra Satelit','overlay':False}
osm={'tiles':'OpenStreetMap','name':'OpenStreetMap','overlay':False}
folium.TileLayer(**(satellite if basemap=='Citra Satelit' else osm)).add_to(m); folium.TileLayer(**(osm if basemap=='Citra Satelit' else satellite)).add_to(m)
for cid,info in CLASSES.items():
    if not enabled[cid]: continue
    rgba=np.zeros((raster.shape[0],raster.shape[1],4),dtype=np.uint8); hx=info['color'].lstrip('#'); rgb=tuple(int(hx[j:j+2],16) for j in (0,2,4)); mask=raster==cid
    rgba[mask,0]=rgb[0]; rgba[mask,1]=rgb[1]; rgba[mask,2]=rgb[2]; rgba[mask,3]=int(255*opacity/100)
    if nodata is not None: rgba[raster==nodata,3]=0
    buff=BytesIO(); Image.fromarray(rgba).save(buff,format='PNG',optimize=True); uri='data:image/png;base64,'+base64.b64encode(buff.getvalue()).decode('ascii')
    folium.raster_layers.ImageOverlay(image=uri,bounds=[[south,west],[north,east]],name=info['name'],opacity=1,interactive=False,show=True).add_to(m)

if show_samples and geo_path is not None:
    try:
        with open(geo_path,encoding='utf-8') as f: gj=json.load(f)
        features=gj.get('features',[]); keys=set()
        for feat in features[:20]: keys.update((feat.get('properties') or {}).keys())
        field=next((k for k in ['class','Class','kelas','label','class_name'] if k in keys),None)
        if field:
            for cid,info in CLASSES.items():
                if not enabled[cid]: continue
                chosen=[]
                for feat in features:
                    label=str((feat.get('properties') or {}).get(field,'')).strip().lower()
                    if label=='fram land': label='farm land'
                    if label==info['label']: chosen.append(feat)
                if chosen:
                    fg=folium.FeatureGroup(name=f'Sampel — {info["name"]}',show=True)
                    folium.GeoJson({'type':'FeatureCollection','features':chosen},style_function=lambda feat,c=info['color']:{'color':c,'weight':2,'fillColor':c,'fillOpacity':sample_opacity/100},tooltip=folium.GeoJsonTooltip(fields=[field],aliases=['Kelas:'])).add_to(fg); fg.add_to(m)
        else: st.warning('Kolom label kelas tidak ditemukan pada GeoJSON.')
    except Exception as e: st.warning(f'Sampel GeoJSON tidak dapat dibaca: {e}')
elif show_samples and geo_path is None: st.info('GeoJSON tidak ditemukan; raster tetap dapat ditampilkan.')
folium.LayerControl(collapsed=True).add_to(m); m.fit_bounds([[south,west],[north,east]])
map_col,help_col=st.columns([3.4,1])
with map_col: st_folium(m,use_container_width=True,height=600,returned_objects=[])
with help_col:
    st.markdown('#### Petunjuk'); st.markdown('- Checkbox sidebar: on/off tiap kelas.\n- Slider: atur transparansi.\n- LayerControl: pilih basemap dan layer sampel.\n- Scroll: zoom peta.')
    st.markdown('<div class="note"><b>Catatan</b><br>Raster diperkecil untuk visualisasi web. File GeoTIFF asli tidak diubah.</div>',unsafe_allow_html=True)

st.markdown('<div class="section">📊 Evaluasi model Random Forest</div>',unsafe_allow_html=True)
st.caption('Metrik berikut berasal dari hasil validasi GEE yang sebelumnya dibagikan, bukan pengukuran ulang dari seluruh piksel raster.')
cm=pd.DataFrame(CM,index=names,columns=names); cm.index.name='Aktual / Prediksi'; left,right=st.columns([1.2,1])
with left:
    st.markdown('#### Confusion matrix'); st.dataframe(cm,use_container_width=True)
    st.download_button('⬇️ Unduh confusion matrix CSV',cm.to_csv().encode('utf-8-sig'),'confusion_matrix_lulc.csv','text/csv')
with right:
    tp=np.diag(CM); actual=CM.sum(axis=1); predicted=CM.sum(axis=0); producer=np.divide(tp,actual,out=np.zeros(5,dtype=float),where=actual!=0); user=np.divide(tp,predicted,out=np.zeros(5,dtype=float),where=predicted!=0)
    metrics=pd.DataFrame({'Kelas':names,'Producer Accuracy (%)':np.round(producer*100,2),'User Accuracy (%)':np.round(user*100,2),'Sampel aktual':actual,'Prediksi':predicted})
    st.markdown('#### Akurasi per kelas'); st.dataframe(metrics,use_container_width=True,hide_index=True)
    st.download_button('⬇️ Unduh akurasi per kelas CSV',metrics.to_csv(index=False).encode('utf-8-sig'),'akurasi_per_kelas_lulc.csv','text/csv')
st.markdown('<div class="section">🌱 Distribusi kelas pada raster</div>',unsafe_allow_html=True)
area=pd.DataFrame([{'Kelas':CLASSES[i]['name'],'Jumlah piksel':counts[i],'Proporsi (%)':round(counts[i]/valid_count*100,2) if valid_count else 0} for i in CLASSES])
st.dataframe(area,use_container_width=True,hide_index=True); st.bar_chart(area.set_index('Kelas')['Proporsi (%)'],use_container_width=True)
st.download_button('⬇️ Unduh distribusi kelas CSV',area.to_csv(index=False).encode('utf-8-sig'),'distribusi_kelas_lulc.csv','text/csv')
st.markdown('<div class="note"><b>Interpretasi:</b> Overall Accuracy adalah proporsi sampel validasi yang diprediksi benar. Producer Accuracy menunjukkan seberapa banyak sampel aktual suatu kelas terdeteksi, sedangkan User Accuracy menunjukkan ketepatan prediksi untuk kelas tersebut. Validasi acak dapat terlalu optimistis bila sampel berdekatan secara spasial.</div>',unsafe_allow_html=True)
st.caption('GeoVision · LULC Jawa Timur · Sentinel-2 SR Harmonized + Random Forest')
