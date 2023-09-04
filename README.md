# 🦋 ChainFury Retrieval Augmented Generation (RAG)

Welcome to the ChainFury Retrieval Augmented Generation (RAG) demo! 🎉

- 🔗 [Click here to try the demo](https://blitzscaling.streamlit.app/)
- 📚 [Access the Blitzscaling PDF](https://drive.google.com/file/d/1QeWwfxEcYyAXkLexCgUX4AWr6nnO3Aqk/view?usp=sharing)

Before we begin, make sure to set the following environment variables:

- 🔸 `OPENAI_TOKEN` - Your OpenAI Token ("sk-xxxxx")
- 🔸 `QDRANT_API_KEY` - Your QDRANT API Key ("hbl-xxxxxx")
- 🔸 `QDRANT_API_URL` - The URL for the QDRANT API ("https://xxx")

## Step One: Loading the data on qdrant

To load the data on qdrant, follow these steps:

1. Install the necessary requirements by running the following command:

   ```bash
   pip install -r requirements_dev.txt
   python3 load_data.py --help
   ```

## Step Two: Running the streamlit app

To run the streamlit app, follow these steps:

1. Install the required dependencies (ignore this step if you already have the dev requirements installed) by running:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the streamlit app using the following command:
   ```bash
   streamlit run streamlit_app.py
   ```

That's it! You're all set to explore the power of ChainFury RAG using the provided demo. 🚀
