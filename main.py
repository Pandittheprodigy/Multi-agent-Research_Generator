import os

# Disable CrewAI telemetry BEFORE importing crewai to avoid signal handler errors
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import warnings
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.llms.ollama import Ollama
from serpapi import GoogleSearch
import subprocess

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- ENV SETUP ---
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    st.error("SERPAPI_API_KEY is not set. Please export it as an environment variable.")
    st.stop()

# --- UTILITIES ---
def serpapi_search(query):
    search = GoogleSearch({"q": query, "api_key": SERPAPI_API_KEY})
    results = search.get_dict()
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
        llm=llm
    )

def save_text_to_md(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    st.write(f"-> Output saved to Markdown: {filename}")

# --- LLM SETUP ---
llm = Ollama(
    model="ollama/mistral",
    base_url="http://localhost:11434"
)

# --- STREAMLIT UI ---
st.title("Multi-Agent Research Generator")
st.markdown("Generate research summaries using AI agents. Enter a topic and subtopics to get started.")

# Input for main topic
topic = st.text_input("Enter the main research topic:", placeholder="e.g., Artificial Intelligence")

# Input for subtopics (dynamic list)
st.subheader("Subtopics")
subtopics_input = st.text_area(
    "Enter subtopics (one per line). Leave blank to use defaults.",
    placeholder="Background\nRecent Advances\nChallenges",
    height=100
)
subtopics = [line.strip() for line in subtopics_input.split('\n') if line.strip()]

if not subtopics:
    st.info("No subtopics entered. Using defaults: ['Background', 'Recent Advances', 'Challenges']")
    subtopics = ["Background", "Recent Advances", "Challenges"]

# Checkbox for arXiv search
run_arxiv = st.checkbox("Also search for related arXiv papers?")

# Button to start research
if st.button("Generate Research"):
    if not topic:
        st.error("Please enter a main research topic.")
    else:
        with st.spinner("Running research agents... This may take a few minutes."):
            try:
                # Create agents
                analyzer = create_agent("Topic Analyzer", topic)
                summarizer = create_agent("Summarizer", topic)
                evaluator = create_agent("Evaluator", topic)

                # Analyze task
                analyze_task = Task(
                    description=f"Break down the topic '{topic}' into 3-5 meaningful subtopics.",
                    agent=analyzer,
                    expected_output="A list of relevant subtopics."
                )

                subtopic_tasks = []
                researcher_tasks = []

                for sub in subtopics:
                    try:
                        links = serpapi_search(sub)
                    except Exception as e:
                        st.warning(f"Error fetching SerpAPI results for '{sub}': {e}")
                        links = []

                    formatted_links = "\n".join(f"{idx+1}. {url}" for idx, url in enumerate(links)) if links else "No links found via SerpAPI."

                    searcher = create_agent("Searcher", sub)
                    search_task = Task(
                        description=(
                            f"The following links were fetched via SerpAPI for subtopic '{sub}':\n"
                            f"{formatted_links}\n\n"
                            "Please review these links and confirm they are relevant, "
                            "or suggest up to 2 additional sources if needed."
                        ),
                        agent=searcher,
                        depends_on=[analyze_task],
                        expected_output="A vetted list of the top 3-5 relevant links."
                    )

                    researcher = create_agent("Researcher", sub)
                    research_task = Task(
                        description=(
                            f"Here is the list of vetted links for subtopic '{sub}':\n"
                            f"{formatted_links}\n\n"
                            "Please extract 3-5 bullet points summarizing the key findings or insights from these sources."
                        ),
                        agent=researcher,
                        depends_on=[search_task],
                        expected_output="3-5 concise bullet points summarizing the research.",
                        markdown=True
                    )

                    subtopic_tasks.extend([search_task, research_task])
                    researcher_tasks.append(research_task)

                # Summary and evaluation tasks
                summary_task = Task(
                    description="Combine all subtopic research outputs into a cohesive 2-3 paragraph summary for the main topic.",
                    agent=summarizer,
                    depends_on=researcher_tasks,
                    expected_output="A cohesive 2-3 paragraph research summary.",
                    markdown=True
                )

                evaluation_task = Task(
                    description="Evaluate the credibility of each source URL provided by the Researcher tasks and assign a score from 1-5.",
                    agent=evaluator,
                    depends_on=researcher_tasks,
                    expected_output="A JSON-style dictionary mapping each URL to a credibility score (1-5)."
                )

                # Run the crew
                crew = Crew(
                    tasks=[analyze_task] + subtopic_tasks + [summary_task, evaluation_task],
                    verbose=True
                )

                results = crew.kickoff()

                # Display results
                st.success("Research completed!")

                # Show summary
                st.subheader("Research Summary")
                st.markdown(str(summary_task.output))

                # Show evaluation
                st.subheader("Source Credibility Evaluation")
                st.json(evaluation_task.output)

                # Save and offer downloads for researcher outputs
                st.subheader("Detailed Research Outputs")
                for task in researcher_tasks:
                    try:
                        output_text = str(task.output)
                        subtopic_safe = task.agent.goal.replace("Assist in generating information for: ", "").replace(" ", "_")
                        md_filename = f"{subtopic_safe}_research_output.md"
                        save_text_to_md(output_text, md_filename)
                        st.download_button(
                            label=f"Download {subtopic_safe} Research",
                            data=output_text,
                            file_name=md_filename,
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error saving Markdown for {task.agent.role}: {e}")

                # Handle arXiv search
                if run_arxiv:
                    st.subheader("ArXiv Papers")
                    try:
                        result = subprocess.run(
                            ["python", "paper.py"],
                            input=f"{topic}\n",
                            text=True,
                            capture_output=True
                        )
                        st.text(result.stdout)
                        if result.stderr:
                            st.warning(f"ArXiv script stderr: {result.stderr}")
                    except Exception as e:
                        st.error(f"Error running ArXiv script: {e}")

            except Exception as e:
                st.error(f"An error occurred during research generation: {e}")
