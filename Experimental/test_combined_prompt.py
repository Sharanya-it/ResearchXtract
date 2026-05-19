import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

LLM_CALL_COUNT = 0

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
def safe_llm_call(prompt, source="unknown"):
    global LLM_CALL_COUNT

    LLM_CALL_COUNT += 1
    print(f"\n🔹 LLM CALL #{LLM_CALL_COUNT} | Source: {source}")

    try:
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        print(f"❌ LLM ERROR from {source}: {e}")
        raise e


def analyze_section_combined(title, content, max_retries=1, retry_delay=3):
    # Optional: trim content during testing so prompts stay lighter
    content = content[:5000]

    prompt = (
        f"You are analyzing the section '{title}' from a research paper.\n\n"

        "Perform BOTH of the following tasks carefully.\n\n"

        "TASK 1: SUMMARY\n"
        "1. Provide a concise overview of the section.\n"
        "2. Identify 2-3 most CRITICAL points or findings.\n"
        "3. Wrap these critical points in '===' "
        "(example: ===This is a key finding===).\n\n"

        "TASK 2: KEY CONCEPTS\n"
        "1. Extract exactly 5 key concepts from the section.\n"
        "2. For each concept, give a short clear definition based only on the content.\n"
        "3. Choose meaningful concepts, not generic words.\n\n"

        "Return your answer in exactly this format:\n\n"

        "SUMMARY:\n"
        "<summary here>\n\n"

        "KEY CONCEPTS:\n"
        "1. Concept Name: Definition\n"
        "2. Concept Name: Definition\n"
        "3. Concept Name: Definition\n"
        "4. Concept Name: Definition\n"
        "5. Concept Name: Definition\n\n"

        f"Content:\n{content}"
    )

    print(f"Prompt length (chars): {len(prompt)}")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}...")
            return safe_llm_call(prompt, "combined_section_analysis")

        except Exception as e:
            last_error = str(e)
            print(f"Attempt {attempt} failed: {last_error}")

            # Retry only for temporary server overload / availability issues
            if "503" in last_error or "UNAVAILABLE" in last_error:
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                    continue

            return f"Combined analysis failed: {last_error}"

    return f"Combined analysis failed after {max_retries} attempts: {last_error}"


if __name__ == "__main__":
    title = "I. INTRODUCTION"
    content = """
The rapid advancement of digital technologies, coupled with the near-universal adoption of mobile
messaging platforms, necessitates a transformation
in how educational institutions handle administrative processes. Among the most recurring yet often
overlooked challenges in large college campuses is
the management of lost and found items. Traditional systems, relying heavily on manual registers,
physical visits to administrative offices, and bulletin
board notices, are fundamentally inefficient and illequipped to handle the volume and velocity of daily
interactions within a modern university environment.
The drawbacks of these conventional approaches are multifold. First, they are timeconsuming for both the users reporting the loss
and the administrators responsible for recordkeeping and matching. Second, they suffer from
data inconsistency and fragmentation, as
records are often scattered across departments or
poorly digitized, leading to missed matches and permanent loss of valuable belongings. Third, they lack
real-time communication, resulting in significant delays between an item being found and its
owner being notified. This often causes frustration
and distrust in the institutional process.
To overcome these limitations, an automated, intelligent solution is essential. This paper proposes
the design and implementation of an AI-Based
Lost and Found System that capitalizes on the
ubiquitous nature of WhatsApp as the primary
user interface. By integrating the system with the
robust Twilio API, we establish a seamless, twoway communication channel, eliminating the need
for users to download or register on separate applications.
The core innovation of our approach lies in the
integration of Artificial Intelligence (AI), specifically Natural Language Processing (NLP) techniques. This allows the system to intelligently interpret unstructured text messages from users (e.g.,
"I lost my blue water bottle near the library on
Monday"), extract key attributes (item name, color,
location, and date), and convert them into structured database entries. The subsequent semantic matching algorithm then compares these structured reports to identify potential matches with
high accuracy, far surpassing the capabilities of
manual keyword filtering.
Furthermore, recognizing the sensitive nature of
user contact details and item descriptions, the system incorporates a Data Encryption and Security
Module utilizing an AES-RSA hybrid framework.
This ensures that all information remains confidential and tamper-proof throughout its storage and
transmission lifecycle.
The long-term objective of this research is to create a reliable, scalable, and user-centric platform
that can be adopted across various high-traffic environments—including public transport hubs, corporate offices, and event venues—beyond the educational context. The subsequent sections of this paper detail the related work, the proposed methodologies for AI and system architecture, the implementation results, and a discussion of the system’s
impact and future directions.
"""

    result = analyze_section_combined(title, content)
    print("\n===== OUTPUT =====\n")
    print(result)
    print(f"\nTotal LLM calls this run: {LLM_CALL_COUNT}")