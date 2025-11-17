"""Example script demonstrating the fact-checking system."""

import asyncio
from agents.orchestrator import create_orchestrator
from utils.logger import log
from agents.reporter_agent import ReporterAgent


async def main():
    """Run example fact-checks."""

    # Example claims to verify
    example_claims = [
        "The COVID-19 vaccine was developed in less than a year using mRNA technology.",
        "The Eiffel Tower is 300 meters tall and located in Paris, France.",
        "Scientists recently discovered that drinking 10 glasses of water daily cures all diseases.",
        "Climate change is causing global temperatures to rise at an unprecedented rate.",
    ]

    # Create orchestrator (without LLM for this example)
    log.info("Initializing fact-checking orchestrator...")
    orchestrator = create_orchestrator(llm_client=None, retriever=None)

    # Create reporter for generating reports
    reporter = ReporterAgent()

    print("\n" + "="*80)
    print("🔍 MULTI-AGENT MISINFORMATION DETECTION SYSTEM - DEMO")
    print("="*80 + "\n")

    # Process each claim
    for i, claim in enumerate(example_claims, 1):
        print(f"\n{'='*80}")
        print(f"CLAIM {i}/{len(example_claims)}")
        print(f"{'='*80}")
        print(f"\n📝 Claim: \"{claim}\"\n")

        try:
            # Run fact-check
            print("⏳ Processing... (this may take a moment)\n")
            result = await orchestrator.check_claim(claim)

            # Display results
            print("✅ Analysis Complete!\n")
            print(f"📊 VERDICT: {result.final_verdict.value}")
            print(f"🎯 CONFIDENCE: {result.confidence:.1%}")
            print(f"⏱️  PROCESSING TIME: {result.processing_time_ms}ms")

            # Show classification
            if result.classification:
                print(f"\n📂 Classification:")
                print(f"   - Theme: {result.classification.theme}")
                print(f"   - Complexity: {result.classification.complexity}/10")
                print(f"   - Urgency: {result.classification.urgency}/10")

            # Show assertions
            print(f"\n🔬 Assertions ({len(result.decomposed_assertions)}):")
            for j, assertion in enumerate(result.decomposed_assertions, 1):
                print(f"   {j}. {assertion.text[:100]}...")
                print(f"      → {assertion.verdict.value} (confidence: {assertion.confidence:.1%})")

            # Show evidence summary
            print(f"\n📰 Evidence: {len(result.evidence_retrieved)} sources collected")

            # Show reasoning trace
            print(f"\n🔄 Processing Steps:")
            for step in result.reasoning_trace:
                print(f"   • {step}")

            # Generate full report
            print(f"\n📄 Generating detailed report...")
            report = reporter.generate_json_report(result)

            # Save report
            import json
            filename = f"report_{result.claim_id}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Report saved to: {filename}")

        except Exception as e:
            print(f"❌ Error processing claim: {str(e)}")
            log.error(f"Error in example script: {str(e)}")

    print("\n" + "="*80)
    print("✅ Demo completed!")
    print("="*80 + "\n")

    # Display workflow visualization
    print("\n📊 WORKFLOW VISUALIZATION:")
    print(orchestrator.get_workflow_visualization())


if __name__ == "__main__":
    asyncio.run(main())
