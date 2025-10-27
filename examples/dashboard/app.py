#!/usr/bin/env python
"""
Main Streamlit application for the PersonaSafe dashboard.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px

# Add project root to path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import PersonaSafe components
from personasafe.core.persona_extractor import PersonaExtractor
from personasafe.core.vector_cache import VectorCache
from personasafe.screening.data_screener import DataScreener
from personasafe.steering.activation_steerer import ActivationSteerer

# Mock data for local UI development
# In the real app, this will come from the screener
def get_mock_screened_data():
    data = {
        'text': [
            'This is a very helpful and kind statement.',
            'I absolutely hate this, it is the worst.',
            'You are probably right, I agree with you.',
            'This is dangerous and should not be done.'
        ],
        'helpful_score': [0.85, -0.4, 0.2, -0.7],
        'toxic_score': [-0.9, 0.95, -0.1, 0.8],
        'sycophantic_score': [0.1, -0.2, 0.9, -0.3],
    }
    return pd.DataFrame(data)

def render_overview_page():
    """Renders the overview and instructions page."""
    st.title("🛡️ PersonaSafe Dashboard")
    st.markdown("""
    Welcome to PersonaSafe, a toolkit for monitoring and screening language model personas.
    
    **Select a page from the sidebar to get started:**
    - **Vector Library:** Manage persona vectors.
    - **Data Screening:** Upload and analyze a dataset for personality drift.
    - **Live Steering:** Generate and compare original vs. steered outputs.
    - **Model Analysis:** (Coming Soon) Explore and compare persona vectors for different models.
    """)

def render_screening_page():
    """Renders the data screening page."""
    st.title("🔍 Data Screening")

    st.sidebar.header("Screening Options")
    
    # --- Model and Trait Selection ---
    available_models = [
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        # Add more models as they are supported
    ]
    selected_model = st.sidebar.selectbox("Select Model", available_models)

    # TODO: In the future, get traits from the vector cache
    available_traits = ["helpful", "toxic", "sycophantic", "honest", "harmful"]
    selected_traits = st.sidebar.multiselect("Select Traits to Screen For", available_traits, default=["toxic", "sycophantic"])

    uploaded_file = st.sidebar.file_uploader("Upload Dataset (.jsonl)", type=['jsonl'])
    run_button = st.sidebar.button("Run Analysis", disabled=(not uploaded_file or not selected_traits))

    # --- Main Page Content ---
    if run_button:
        st.write("### Screening Results")
        
        # Read the uploaded file
        df = pd.read_json(uploaded_file, lines=True)
        st.write(f"Loaded {len(df)} samples from {uploaded_file.name}")

        with st.spinner(f"Running analysis with {selected_model}..."):
            try:
                # 1. Initialize the extractor
                # This will use the cache for persona vectors
                extractor = PersonaExtractor(model_name=selected_model)
                
                # 2. Load the required persona vectors from cache
                # Vectors must be pre-computed using scripts/quick_demo.py or extraction pipeline
                st.write("Loading persona vectors from cache...")
                cache = VectorCache()
                persona_vectors = {}
                missing_vectors = []

                for trait in selected_traits:
                    # Try to load from cache
                    vector = cache.get(selected_model, trait)
                    if vector is not None:
                        persona_vectors[trait] = vector
                        st.success(f"✓ Loaded {trait} vector from cache")
                    else:
                        missing_vectors.append(trait)

                # If any vectors are missing, show helpful error
                if missing_vectors:
                    st.error(f"⚠️ Missing pre-computed vectors for: {', '.join(missing_vectors)}")
                    st.info("""
                    **To generate persona vectors, run:**
                    ```bash
                    python scripts/quick_demo.py --model {model} --trait {trait}
                    ```
                    Or see docs/internal/GUIDES/03_HPC_GUIDE.md for batch extraction on HPC.
                    """.format(model=selected_model, trait=missing_vectors[0]))
                    st.stop()  # Don't continue if vectors are missing

                # 3. Initialize the screener
                screener = DataScreener(extractor=extractor, persona_vectors=persona_vectors)

                # 4. Screen the dataset
                st.write("Scoring dataset...")
                screened_df = screener.screen_dataset(df, text_column="text") # Assumes 'text' column

                # 5. Generate the report
                st.write("Generating report...")
                report = screener.generate_report(screened_df)

                st.success("Analysis Complete!")

                # Display results
                st.write("### Screened Data")
                st.dataframe(screened_df)

                # CSV export
                st.download_button(
                    label="Download scores as CSV",
                    data=screened_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"screened_{selected_model.replace('/', '_')}.csv",
                    mime="text/csv",
                )

                st.write("### Summary Report")
                st.json(report)

                # Plot histograms per trait
                score_cols = [c for c in screened_df.columns if c.endswith("_score")]
                if score_cols:
                    st.write("### Score Distributions")
                    tabs = st.tabs([c.replace("_score", "") for c in score_cols])
                    for tab, col in zip(tabs, score_cols):
                        with tab:
                            fig = px.histogram(screened_df, x=col, nbins=30, title=f"Distribution of {col}")
                            st.plotly_chart(fig, use_container_width=True)

                # Optional trait correlation heatmap (if multiple traits)
                if len(score_cols) > 1:
                    st.write("### Trait Correlation Heatmap")
                    corr = screened_df[score_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Trait Correlations")
                    st.plotly_chart(fig, use_container_width=True)

                # HPC helper commands
                with st.expander("Run on HPC (commands)"):
                    helper = f"""
                    # Prepare screening job locally
                    python scripts/hpc_bridge.py prepare-screening \
                      --model {selected_model} \
                      --dataset {uploaded_file.name} \
                      --traits {' '.join(selected_traits)} \
                      --out-dir jobs

                    # rsync job to HPC, run, and sync results back (example paths)
                    rsync -avz jobs/ $USER@hpc:/path/to/jobs/
                    # ... submit job on HPC ...
                    rsync -avz $USER@hpc:/path/to/jobs/reports/ ./reports/
                    """
                    st.code(helper, language="bash")

            except Exception as e:
                st.error(f"An error occurred during analysis:")
                st.exception(e)

    else:
        st.info("To begin, select your desired model and traits, upload a dataset, and click 'Run Analysis'.")

def render_analysis_page():

    """Renders the model analysis page."""

    st.title("📊 Model Analysis")

    st.warning("This page is under construction.")



def render_steering_page():

    """Renders the live activation steering page."""

    st.title("🧠 Live Activation Steering")

    st.markdown("""

    Interactively steer model generations in real-time. This requires running on a machine 

    with the model and persona vectors loaded (e.g., the HPC).

    """)



    # --- Steer Options ---

    st.sidebar.header("Steering Options")

    prompt = st.sidebar.text_area("Prompt", "The meaning of life is")



    available_traits = ["helpful", "toxic", "sycophantic", "honest", "harmful"]

    trait_to_steer = st.sidebar.selectbox("Trait to Steer", available_traits)



    multiplier = st.sidebar.slider(

        "Steering Multiplier", 

        min_value=-3.0, 

        max_value=3.0, 

        value=1.5, 

        step=0.1

    )



    # Presets and generation params
    st.sidebar.markdown("**Presets & Generation**")
    preset = st.sidebar.selectbox("Preset", ["None", "Helpful", "Toxic", "Honest"], index=0)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
    max_tokens = st.sidebar.slider("Max New Tokens", min_value=16, max_value=256, value=128, step=16)
    seed = st.sidebar.number_input("Seed (optional)", value=0, step=1)

    run_steering = st.sidebar.button("Generate Steered Output")



    # --- Main Page Content ---

    if run_steering:

        st.write("### Steering Results")

        col1, col2 = st.columns(2)

        

        with col1:

            st.subheader("Original Output")

            with st.spinner("Generating original text..."):
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch
                    model_id = "google/gemma-3-4b"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
                    steerer = ActivationSteerer(model=model, tokenizer=tokenizer)
                    zero_vec = torch.zeros(model.config.hidden_size)
                    outputs = steerer.steer(
                        prompt=prompt,
                        persona_vector=zero_vec,
                        multiplier=0.0,
                        layer=max(0, model.config.num_hidden_layers // 2),
                        max_new_tokens=int(max_tokens),
                    )
                    st.success("Original Output")
                    st.write(outputs[0])
                except Exception as e:
                    st.error("Failed to generate original output")
                    st.exception(e)

        

        with col2:

            st.subheader(f"Steered towards `{trait_to_steer}` (x{multiplier})")

            with st.spinner("Generating steered text..."):
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch
                    model_id = "google/gemma-3-4b"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
                    steerer = ActivationSteerer(model=model, tokenizer=tokenizer)
                    cache = VectorCache()
                    vec = cache.get(model_id, trait_to_steer)
                    if vec is None:
                        st.warning(f"Trait vector '{trait_to_steer}' not found in cache. Using zero vector (no steering).")
                        vec = torch.zeros(model.config.hidden_size)
                    # Apply preset
                    if preset.lower() in ["helpful", "toxic", "honest"]:
                        trait_to_use = preset.lower()
                        cached_vec = cache.get(model_id, trait_to_use)
                        if cached_vec is not None:
                            vec = cached_vec
                            st.info(f"Preset '{preset}' using cached vector")
                    outputs = steerer.steer(
                        prompt=prompt,
                        persona_vector=vec,
                        multiplier=float(multiplier),
                        layer=max(0, model.config.num_hidden_layers // 2),
                        max_new_tokens=int(max_tokens),
                    )
                    st.success("Steered Output")
                    st.write(outputs[1])

                    # Save run artifact for later review
                    try:
                        from pathlib import Path
                        Path("runs").mkdir(exist_ok=True)
                        run_record = {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "model": model_id,
                            "prompt": prompt,
                            "trait": trait_to_steer,
                            "multiplier": float(multiplier),
                            "layer": int(max(0, model.config.num_hidden_layers // 2)),
                            "original": outputs[0],
                            "steered": outputs[1],
                        }
                        out_path = Path("runs") / f"run_{int(datetime.utcnow().timestamp())}.json"
                        with out_path.open("w") as f:
                            json.dump(run_record, f, ensure_ascii=False, indent=2)
                        st.info(f"Saved run to {out_path}")
                    except Exception as e:
                        st.warning("Could not save run artifact")
                        st.exception(e)
                except Exception as e:
                    st.error("Failed to generate steered output")
                    st.exception(e)

    else:

        st.info("Enter a prompt and click 'Generate Steered Output' to see the results.")

    # HPC helper commands
    with st.expander("Run on HPC (commands)"):
        helper = f"""
        # Prepare steering job locally
        python scripts/hpc_bridge.py prepare-steering \
          --model google/gemma-3-4b \
          --prompt "{prompt}" \
          --trait {trait_to_steer} \
          --multiplier {float(multiplier)} \
          --layer 20 \
          --out-dir jobs

        # rsync job to HPC / run / sync back
        rsync -avz jobs/ $USER@hpc:/path/to/jobs/
        # ... submit job on HPC ...
        rsync -avz $USER@hpc:/path/to/jobs/runs/ ./runs/
        """
        st.code(helper, language="bash")


def render_vector_library_page():
    """Renders the persona vector library page."""
    st.title("📚 Vector Library")

    cache = VectorCache()

    st.subheader("Cached Vectors")
    cached = cache.list_cached_as_list()
    if cached:
        st.dataframe(pd.DataFrame(cached))
    else:
        st.info("No cached vectors found. Generate via scripts or upload below.")

    st.subheader("Add Vector (.pt)")
    with st.form("upload_vector_form"):
        model_name = st.text_input("Model name", value="google/gemma-3-4b")
        trait_name = st.text_input("Trait name", value="helpful")
        file = st.file_uploader("Vector file (.pt)", type=["pt"])
        submitted = st.form_submit_button("Add to Cache")
        if submitted:
            if not (model_name and trait_name and file):
                st.error("Please provide model name, trait name, and a .pt file.")
            else:
                try:
                    import torch
                    from io import BytesIO
                    tensor = torch.load(BytesIO(file.read()))
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.tensor(tensor)
                    cache.set(model_name, trait_name, tensor)
                    st.success(f"Added vector for {model_name}/{trait_name}")
                except Exception as e:
                    st.error("Failed to add vector to cache")
                    st.exception(e)


def render_runs_viewer_page():
    """Browse saved runs from runs/*.json and re-render details."""
    st.title("🗂️ Runs Viewer")
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    run_files = sorted(runs_dir.glob("run_*.json"), reverse=True)
    if not run_files:
        st.info("No saved runs found.")
        return
    st.sidebar.header("Filter Runs")
    trait_filter = st.sidebar.text_input("Trait contains", value="")
    model_filter = st.sidebar.text_input("Model contains", value="")
    filtered = []
    for f in run_files:
        try:
            with f.open("r") as fh:
                data = json.load(fh)
            if trait_filter and trait_filter.lower() not in str(data.get("trait", "")).lower():
                continue
            if model_filter and model_filter.lower() not in str(data.get("model", "")).lower():
                continue
            filtered.append((f, data))
        except Exception:
            continue
    if not filtered:
        st.info("No runs match the current filters.")
        return
    for f, data in filtered[:200]:
        with st.expander(f.name):
            st.json(data)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Output")
                st.write(data.get("original", ""))
            with col2:
                st.subheader("Steered Output")
                st.write(data.get("steered", ""))

def main():

    """Main function to run the Streamlit app."""

    st.set_page_config(

        page_title="PersonaSafe",

        page_icon="🛡️",

        layout="wide"

    )



    # Sidebar navigation

    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["Overview", "Vector Library", "Data Screening", "Live Steering", "Runs Viewer", "Model Analysis"])



    if page == "Overview":

        render_overview_page()

    elif page == "Vector Library":

        render_vector_library_page()

    elif page == "Data Screening":

        render_screening_page()

    elif page == "Live Steering":

        render_steering_page()

    elif page == "Runs Viewer":

        render_runs_viewer_page()

    elif page == "Model Analysis":

        render_analysis_page()

if __name__ == "__main__":
    main()
