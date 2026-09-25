"""
Prompt template + LLM call + structured output parsing.
Turns (JD text + retrieved resume context) into a validated ResumeEvaluation.
"""

from dotenv import load_dotenv

load_dotenv()  # must run before ChatOpenAI() is created below, so OPENAI_API_KEY is set

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

try:
    from .schema import ResumeEvaluation  # when imported as screener.chain (e.g. from app.py)
except ImportError:
    from schema import ResumeEvaluation  # when run directly as: python screener/chain.py

LLM_MODEL = "gpt-4o-mini"

parser = PydanticOutputParser(pydantic_object=ResumeEvaluation)

PROMPT_TEMPLATE = """You are an experienced, fair recruiter evaluating a candidate's resume against a specific job description.

STRICT RULES:
- Base your evaluation ONLY on the resume text provided below. Do not assume, infer, or invent skills, experience, or qualifications that are not explicitly stated in the resume text.
- Score against explicit criteria implied by the job description: required skills, relevant experience, education, and relevant projects. Do not give a high score just because the resume looks generally strong; it must match THIS job description specifically.
- Be honest about gaps and mismatches. If the resume shows unrelated or only tangential experience, reflect that clearly and proportionally in the score and recommendation. A resume for a completely different field should score low, not be graded on general competence.
- If the job description is too vague or short to identify clear requirements, set insufficient_jd_detail to true and explain what's missing in the justification, but still give your best-effort score based on whatever IS specified.
- This is a professional evaluation. Do not exaggerate strengths or soften weaknesses to be encouraging.

JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME EXCERPTS (retrieved as most relevant to this JD):
{resume_context}

CANDIDATE LABEL: {candidate}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# temperature=0 for consistent, repeatable scores across runs on the same input
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

chain = prompt | llm | parser


def evaluate_candidate(candidate: str, jd_text: str, resume_context: str) -> ResumeEvaluation:
    """
    Run the evaluation chain for one candidate and return a validated
    ResumeEvaluation object.
    """
    result = chain.invoke({
        "candidate": candidate,
        "jd_text": jd_text,
        "resume_context": resume_context,
        "format_instructions": parser.get_format_instructions(),
    })
    return result


if __name__ == "__main__":
    # Quick manual test - run:
    #   python screener/chain.py "<job description text>" <resume1.pdf> <resume2.pdf> ...
    # This calls the OpenAI chat + embeddings APIs, so it needs OPENAI_API_KEY
    # in .env with active credits.
    import sys
    from ingest import load_and_split_multiple
    from store import (
        build_candidate_stores,
        get_candidates,
        retrieve_for_candidate,
        format_retrieved_context,
    )

    if len(sys.argv) < 3:
        print("Usage: python chain.py '<job description text>' <resume1.pdf> <resume2.pdf> ...")
        sys.exit(1)

    jd_text = sys.argv[1]
    pdf_paths = sys.argv[2:]

    print("Loading and splitting resumes...")
    chunks = load_and_split_multiple(pdf_paths)

    print("Building per-candidate FAISS stores...")
    stores = build_candidate_stores(chunks)

    for candidate in get_candidates(stores):
        print(f"\n=== Evaluating {candidate} ===")
        docs = retrieve_for_candidate(stores, candidate, jd_text, k=5)
        context = format_retrieved_context(docs)
        result = evaluate_candidate(candidate, jd_text, context)
        print(result.model_dump_json(indent=2))