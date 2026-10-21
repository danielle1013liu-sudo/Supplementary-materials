import base64
import os
import uuid
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data import ROOT, file_hash, read_json
from src.hydrogel_immune_ml.inference import load_resources, predict_form
from src.hydrogel_immune_ml.public_encoding import fields

st.set_page_config(page_title=os.getenv('HYDROGEL_APP_NAME', 'Hydrogel Cytokine Explorer'), layout='wide', initial_sidebar_state='collapsed')
st.markdown('<style>[data-testid="stHeader"],footer{display:none} .stMainBlockContainer{padding:0!important;max-width:none!important} .stApp{background:#F6F0F6} [data-testid="stElementContainer"]{margin:0} [data-testid="stVerticalBlock"]{gap:0!important} iframe{display:block;border:0}</style>', unsafe_allow_html=True)


@st.cache_resource
def resources(release_key, runtime_profile):
    return load_resources(runtime=runtime_profile)


def file_uri(path, mime):
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


runtime_profile = os.getenv('HYDROGEL_RUNTIME_PROFILE', 'reproduction')
release_key = (runtime_profile,) + tuple(file_hash(ROOT / name) for name in (
    'data/checksums.json', 'models/manifest.json', 'data/pareto/CURRENT_RELEASE.json',
))
if st.session_state.get('release_key') != release_key:
    st.session_state.release_key = release_key
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.form = None
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.last_event = None

numeric = read_json(ROOT / 'data/feature_schema.json')['numeric_semantics']
ranges = pd.read_csv(ROOT / 'results/revision_current/numeric_display_ranges.csv').set_index('field')
for field, details in numeric.items():
    details.update({'minimum': float(ranges.loc[field, 'minimum']), 'maximum': float(ranges.loc[field, 'maximum'])})

component = components.declare_component('hydrogel_explorer', path=str(ROOT / 'web'))
supplement_path = ROOT / 'docs/Supplementary_IL10_updated_20261020.docx'
event = component(
    name=os.getenv('HYDROGEL_APP_NAME', 'Hydrogel Cytokine Explorer'),
    fields=fields(), numeric=numeric, form=st.session_state.form,
    result=st.session_state.result, error=st.session_state.error,
    banner=file_uri(ROOT / 'assets/banner.svg', 'image/svg+xml'),
    banner_mobile=file_uri(ROOT / 'assets/banner_mobile.svg', 'image/svg+xml') if (ROOT / 'assets/banner_mobile.svg').is_file() else None,
    supplement=file_uri(supplement_path, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
    supplement_filename=supplement_path.name,
    key='hydrogel-' + '-'.join(value[:12] for value in release_key), default=None,
)
if event and event.get('id') != st.session_state.last_event:
    st.session_state.last_event = event['id']
    st.session_state.form = event['form']
    st.session_state.result = None
    st.session_state.error = None
    if event.get('action') == 'predict':
        try:
            destination = ROOT / 'results/app' / st.session_state.session_id / uuid.uuid4().hex
            st.session_state.result = predict_form(event['form'], resources(release_key, runtime_profile), destination)
        except (ValueError, AssertionError) as error:
            st.session_state.error = str(error) or 'The release verification failed.'
    st.rerun()
