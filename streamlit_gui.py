#import shap_edited.shap as shap
import streamlit as st
import streamlit.components.v1 as components
#from fact_check_platform.constants import *
#from fact_check_platform.explain import TextWrapper, explain_shap
#from loaders import (anserini_loader, hybrid_loader, load_explainability,
#                     load_nli_models, load_pipeline, load_pipeline_data,
#                     load_retrievers)
from PIL import Image
from htmls import *

# set application configuration (must be the first st command)
# Loading Image using PIL
im = Image.open('./figures/icon_aic3.jpg')
# Adding Image to web app
st.set_page_config(page_title="Fact-checking platform", page_icon = im, layout="wide")

#configuration dictionary
_CONFIG = {
    "db_path": "/home/mlynatom/data/cs_wiki_revid_db_sqlite.db",
    "db_name": "CsWiki",
    "retrieval": [{"name": "Anserini",
                   #"loader": anserini_loader,
                   "kwargs": {"model_dir": "/home/mlynatom/data/indices/wiki_id","k1": 0.5,"b": 0.6,"language": "cs"}
                   },
                   {"name": "Hybrid",
                    #"loader": hybrid_loader,
                    "kwargs": {"model_dir": "/home/mlynatom/data/indices/wiki_id", "k1": 0.5, "b": 0.6, "language": "cs", 
                               "cross_encoder_path": 'cross-encoder/ms-marco-MiniLM-L-6-v2', "cross_encoder_max_length" : 512}
                    }],
    "nli":[{
                "name": "original",
                "model_path":"/mnt/data/factcheck/fever-cs-fbtranslate-models/deepset/xlm-roberta-large-squad2_bs9_ep20_wr0.4",
                "max_length": 512,
                "scaling_model_checkpoint":"/home/mlynatom/data/temp_checkpoint/lightning_logs/original/checkpoints/epoch=4-step=3420.ckpt"
            },
            {
                "name": "F1",
                "model_path":"/home/mlynatom/xlm-roberta-large-squad2-csfever_v2-f1",
                "max_length": 512,
                "scaling_model_checkpoint":f"/home/mlynatom/xlm-roberta-large-squad2-csfever_v2-f1/temperature.ckpt"
            },
            {
                "name": "precision",
                "model_path":"/home/mlynatom/xlm-roberta-large-squad2-csfever_v2-precision",
                "max_length": 512,
                "scaling_model_checkpoint":"/home/mlynatom/xlm-roberta-large-squad2-csfever_v2-precision/temperature.ckpt"
            },
            {
                "name": "0.7",
                "model_path":"/home/mlynatom/xlm-roberta-large-squad2-csfever_v2-07",
                "max_length": 512,
                "scaling_model_checkpoint":"/home/mlynatom/xlm-roberta-large-squad2-csfever_v2-07/temperature.ckpt"
            },],
    "explainability": {},
}

# init fact_checking pipeline components
#db = load_pipeline_data(_CONFIG)
#retrievers = load_retrievers(_CONFIG, db)
#nli_models, scaling_models = load_nli_models(_CONFIG)
#explainers = load_explainability(_CONFIG, nli_models)
#meta_model = load_pipeline(_CONFIG, db)

retrievers = {"Anserini": "", "Hybrid": ""}
nli_models = {"original": "", "F1": "", "precision": "", "0.7": ""}


# start application
st.image("./figures/logo_aic.png")
st.title("Fact-checking platform Streamlit GUI")

with st.form("my_form"):
    #init columns
    column_1, column_2, column_3, column_4, column_5 = st.columns([3, 1, 1, 1, 1])

    #column 1
    claim = column_1.text_input("Claim", "FEL ČVUT je fakultou ČVUT.")

    #column 2
    num_results = int(column_2.number_input("Number of results", min_value= 1, max_value=10, value=2, step=1))

    #column 3
    retriever_name = str(column_3.selectbox("Retriever", retrievers.keys()))
    nli_model_name = str(column_3.selectbox("NLI model", nli_models.keys()))

    #column 4
    calibrate = column_4.checkbox(
        "Calibrate model using Temperature Scaling", True)
    output_mode = column_4.radio("Output mode", ("Basic", "Explain using SHAP (slow)", "Render Wikipedia (experimental)"))

    #column 5
    search = column_5.form_submit_button("Search")

#action
if search:
    with st.spinner("Computing..."):
        #results = meta_model.retrieve(query=claim, k=num_results, retriever=retrievers[retriever_name], 
        #                              nlimodel=nli_models[nli_model_name], scaling_model=scaling_models[nli_model_name], 
        #                              temp_scaling=calibrate)

        results = [
            {
                "id": 53321,
                "content": """Fakulta elektrotechnická ČVUT (FEL ČVUT) je fakulta ČVUT s cca 3 100 studenty, 730 zaměstnanci a 
                              ročním rozpočtem přesahujícím 800 milionů korun. Poslání fakulty. Elektrotechnická fakulta ČVUT vychovává odborníky 
                              v oblasti elektrotechniky, energetiky, softwarového inženýrství, sdělovací techniky, robotiky a kybernetiky, automatizace, 
                              informatiky a výpočetní techniky. Je také centrem pro vědeckou a výchovnou činnost v uvedených oblastech. 
                              Studijní programy. Fakulta elektrotechnická uskutečňuje výuku ve studijních programech. 
                              V prvním ročníku si již vybírají studenti obor: Věda a výzkum. Fakulta je jedním z největších výzkumných pracovišť v ČR, 
                              (pátým dle aktuálního hodnocení Rady vlády pro výzkum a vývoj). Počítačové studovny. Počítačové studovny s volným přístupem v 
                              Dejvicích provozuje oddělení výpočetní techniky Střediska vědeckotechnických informací (SVTI). Na Karlově náměstí se také nachází 
                              několik místností s počítači se systémem Windows nebo Solaris. Samozřejmostí je možnost Wi-Fi připojení Eduroam. Katedry. Výuka i 
                              výzkum jsou na fakultě organizovány katedrami, tj. specializovanými pracovišti. Katedry fakulty mají přidělený alfanumerický kód K131xx, 
                              jednoznačný v rámci celé univerzity. Symbol "xx" představuje dvojciferné číslo, pod kterým katedra vystupuje v rámci fakulty. 
                              Toto číslo je součástí kódů vyučovaných předmětů, čímž přispívá k jednoznačnému určení předmětu dle kódu. 
                              K 1. srpnu 2007 působilo na fakultě 17 kateder, jedno centrum a jedno středisko. V akademickém roce 2006/2007 zanikla "Katedra 
                              tělesné výchovy". Výuku tělesné výchovy zajišťuje od akademického roku 2007/2008 "Ústav tělesné výchovy a sportu ČVUT" (ÚTVS ČVUT). 
                              Studentská konference POSTER. FEL ČVUT každoročně v květnu pořádá studentskou konferenci POSTER, na které jsou prezentovány výsledky 
                              práce studentů a doktorandů. Tato konference je vynikající příležitostí k setkání s aktivními a profesně zdatnými studenty. Zhruba čtvrtina příspěvků je zahraničních. 
                              Spolek absolventů. Absolventi FEL se sdružují ve spolku ELEKTRA. Spolek pořádá každoroční srazy absolventů, koncerty a další akce.""",
                "html": HTML_1,
                "url": "https://cs.wikipedia.org/wiki?curid=53321",
                "revid": 20979747,
                "title": "Fakulta elektrotechnická ČVUT"
            },
            {
                "id": 458962,
                "content": """Fakulta elektrotechnická Západočeské univerzity v Plzni (FEL) je jednou z devíti fakult Západočeské univerzity v Plzni (ZČU).
                 Historie. V roce 1949 byla založena jako Vysoká škola strojní a elektrotechnická v Plzni (VŠSE) a patřila pod České vysoké učení technické 
                 v Praze. Od roku 1950 fungovala jako samostatná fakulta na VŠSE. Roku 1953 se VŠSE oddělila od ČVUT a roku 1960 se tato vysoká škola 
                 rozdělila na fakulty a tak vznikla Fakulta elektrotechnická. V roce 1991 patřila k zakládajícím institucím Západočeské univerzity v Plzni 
                 (28. září 1991). Činnost fakulty. Výuka. Fakulta nabízí studium s dosažením vysokoškolského titulu bakalář (Bc.), inženýr (Ing.) a doktor 
                 (Ph.D.). Organizační struktura. Děkanem je od 1. března 2018 Zdeněk Peroutka. Katedry a pracoviště. Katedra materiálů a technologií (KET). 
                 Katedra zabezpečuje výuku ve všech programech magisterského a bakalářského studia v oblastech materiálů a technologií pro elektrotechniku a 
                 elektroniku, měření a měřicích systémů a podnikání a řízení průmyslových systémů v elektrotechnice. Z hlediska výzkumu a vývoje se člení na 
                 dva vzájemně se doplňující týmy materiálového výzkumu a diagnostiky. Součástí katedry jsou laboratoře sloužící jak pro pedagogické tak i pro 
                 výzkumné účely. Kromě laboratoří sloužících ke studiu a analýzám materiálů a technologií, zde jsou také akustické laboratoře s dozvukovou a 
                 bezodrazovou komorou a mikroskopová laboratoř, která je regionálním referenčním pracovištěm firmy Olympus Czech Group, s.r.o. 
                 Kromě toho katedra spolupracuje na řešení konkrétních úkolů s externími podniky.""",
                "html": HTML_2,
                "url": "https://cs.wikipedia.org/wiki?curid=458962",
                "revid": 21176988,
                "title": "Fakulta elektrotechnická Západočeské univerzity"
            }
        ]

    for result in results:
        id_ = result["id"]
        if id_ is None:
            continue

        url = result["id"]
        revid = result["revid"]
        title = result["title"]
        old_url = f"https://cs.wikipedia.org/w/index.php?title={title}&oldid={revid}"
        st.markdown(
            f"Id: {id_} **|** **{title}** **|** [Wikipedia page](<{url}>) **|** [Old Wikipedia page](<{old_url}>) **|** **:green[SUPPORTS 94.50]** **|** **:red[REFUTES 1.94]** **|** **:orange[NOT ENOUGH INFO 3.56]**")
        with st.expander("Expand evidence: "):
            content = result["content"]
            if output_mode == "Explain using SHAP (slow)":
                with st.spinner("Explaining..."):
                    #shap_values = explain_shap(TextWrapper(
                    #    [content, claim]), explainers[nli_model_name], nli_models[nli_model_name], verbose=False)
                    #html = shap.plots.text(shap_values, display=False)
                    html = result["html"]
                st.markdown(html, unsafe_allow_html=True)
            elif output_mode=="Render Wikipedia (experimental)":
                components.iframe(src=old_url, height=500, scrolling=True)
            else:
                st.write(content)
