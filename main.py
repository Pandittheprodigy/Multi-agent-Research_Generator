import os
import warnings
import streamlit as st
import subprocess
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI  # Use ChatOpenAI for modern CrewAI
import serpapi  # Correct import for the newer library

# Disable CrewAI telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- ENV SETUP ---
# It is safer to use st.secrets in Streamlit Cloud or local .env
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not SERPAPI_API_KEY or not OPENROUTER_API_KEY:
    st.error("Missing API Keys. Please set SERPAPI_API_KEY and OPENROUTER_API_KEY.")
    st.stop()

# --- LLM SETUP (OpenRouter) ---
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct", 
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
)

# --- UTILITIES ---
def serpapi_search(query):
    # Corrected for 'serpapi' library (Client class)
    client = serpapi.Client(api_key=SERPAPI_API_KEY)
    results = client.search({"q": query, "engine": "google"})
    
    links = []
    for r in results.get("organic_results", []):
        link = r.get("link")
        if link:
            links.append(link)
        if len(links) >= 5:
            break
    return links

def create_agent(role, topic):
    backstory = f"You are a skilled {role.lower()} working on topic '{topic}'."
    return Agent(
        role=role,
        goal=f"Assist in generating information for: {topic}",
        backstory=backstory,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

# --- STREAMLIT UI ---
st.title("Multi-Agent Research Generator")

topic = st.text_input("Enter the main research topic:", placeholder="e.g., Artificial Intelligence")
subtopics_input = st.text_area("Enter subtopics (one per line):", placeholder="Background\nRecent Advances", height=100)

subtopics = [line.strip() for line in subtopics_input.split('\n') if line.strip()]
if not subtopics:
    subtopics = ["Background", "Recent Advances", "Challenges"]

run_arxiv = st.checkbox("Also search for related arXiv papers?")

if st.button("Generate Research"):
    if not topic:
        st.error("Please enter a main research topic.")
    else:
        with st.spinner("Running agents..."):
            try:
                all_tasks = []
                researcher_tasks = []
                
                # 1. Analyze Task
                analyzer = create_agent("Topic Analyzer", topic)
                analyze_task = Task(
                    description=f"Break down '{topic}' into 3-5 sub-themes.",
                    agent=analyzer,
                    expected_output="A list of relevant subtopics."
                )
                all_tasks.append(analyze_task)

                # 2. Dynamic Subtopic Tasks
                for sub in subtopics:
                    links = serpapi_search(f"{topic} {sub}")
                    formatted_links = "\n".join(links) if links else "No links found."

                    researcher = create_agent("Researcher", sub)
                    res_task = Task(
                        description=f"Summarize these links for {sub}:\n{formatted_links}",
                        agent=researcher,
                        expected_output="3-5 bullet points of key insights.",
                    )
                    all_tasks.append(res_task)
                    researcher_tasks.append(res_task)

                # 3. Final Tasks
                summarizer = create_agent("Summarizer", topic)
                summary_task = Task(
                    description="Combine all previous insights into a 3-paragraph summary.",
                    agent=summarizer,
                    expected_output="A cohesive markdown research summary."
                )
                all_tasks.append(summary_task)

                # Run Crew
                crew = Crew(agents=[analyzer, summarizer], tasks=all_tasks, verbose=True)
                crew.kickoff()

                # Display Output
                st.success("Research completed!")
                st.subheader("Final Summary")
                # Accessing output via the new CrewAI TaskOutput object
                st.markdown(summary_task.output.raw)

                for i, task in enumerate(researcher_tasks):
                    with st.expander(f"Details: {subtopics[i]}"):
                        st.write(task.output.raw)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Handle arXiv as a subprocess (ensure paper.py exists in the directory)
if run_arxiv and topic:
    if st.button("Fetch arXiv Papers"):
        subprocess.run(["python", "paper.py", topic])
