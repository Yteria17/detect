"""
Script d'Exemple - Système Multi-Agents de Détection de Désinformation

Ce script démontre comment utiliser le système complet pour vérifier
des affirmations et détecter la désinformation.
"""

import yaml
from pathlib import Path
from datetime import datetime
import json

# Import du workflow
from workflow import MultiAgentFactChecker

# Import des utilitaires
from utils import DeepfakeDetector


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Charge la configuration depuis le fichier YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def example_1_simple_claim():
    """
    Exemple 1: Vérification simple d'une affirmation.
    """
    print("\n" + "="*70)
    print("EXEMPLE 1: Vérification Simple d'une Affirmation")
    print("="*70)

    # Configuration
    config = load_config()

    # Initialisation du système
    fact_checker = MultiAgentFactChecker(
        llm_client=None,  # Pour la démo, pas de LLM réel
        vector_store=None,  # Pas de vector store pour la démo
        config=config.get('agents', {})
    )

    # Affirmation à vérifier
    claim = """
    Le COVID-19 a été créé en laboratoire en 2019 et les vaccins
    contiennent des puces électroniques pour surveiller la population.
    """

    print(f"\n📋 Affirmation à vérifier:\n{claim.strip()}")

    # Vérification
    print("\n🔍 Lancement de la vérification...")
    result = fact_checker.check_claim(claim)

    # Affichage du résultat
    print("\n" + "-"*70)
    print("RÉSULTAT DE LA VÉRIFICATION")
    print("-"*70)

    verdict = result.get('verdict', {})
    print(f"\n✓ Verdict: {verdict.get('verdict_label', 'N/A')}")
    print(f"✓ Confiance: {verdict.get('confidence', 0):.1%}")
    print(f"✓ Explication: {verdict.get('explanation', 'N/A')}")

    # Recommandations
    recommendations = result.get('recommendations', [])
    if recommendations:
        print("\n💡 Recommandations:")
        for rec in recommendations:
            print(f"  • {rec}")

    # Export du rapport
    output_dir = Path("output/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fact_checker.export_report(result, str(report_path), format='json')
    print(f"\n📄 Rapport exporté: {report_path}")


def example_2_multiple_claims():
    """
    Exemple 2: Vérification de plusieurs affirmations.
    """
    print("\n" + "="*70)
    print("EXEMPLE 2: Vérification de Plusieurs Affirmations")
    print("="*70)

    config = load_config()
    fact_checker = MultiAgentFactChecker(
        llm_client=None,
        vector_store=None,
        config=config.get('agents', {})
    )

    # Liste d'affirmations
    claims = [
        "Paris est la capitale de la France.",
        "La Terre est plate et tous les scientifiques mentent.",
        "Les voitures électriques sont 100% écologiques et n'ont aucun impact environnemental."
    ]

    print(f"\n📋 Vérification de {len(claims)} affirmations...\n")

    # Vérification batch
    results = fact_checker.check_multiple_claims(claims)

    # Affichage synthétique
    print("\n" + "-"*70)
    print("RÉSULTATS")
    print("-"*70)

    for i, result in enumerate(results, 1):
        verdict = result.get('verdict', {})
        print(f"\n{i}. {claims[i-1][:60]}...")
        print(f"   → {verdict.get('verdict_label', 'N/A')} ({verdict.get('confidence', 0):.0%})")


def example_3_deepfake_detection():
    """
    Exemple 3: Détection de deepfakes.
    """
    print("\n" + "="*70)
    print("EXEMPLE 3: Détection de Deepfakes")
    print("="*70)

    config = load_config()
    deepfake_config = config.get('deepfake', {})

    detector = DeepfakeDetector(deepfake_config)

    print("\n🎥 Simulation de détection de deepfake vidéo...")

    # Simulation (dans un cas réel, on passerait un vrai fichier)
    video_path = "path/to/video.mp4"

    result = detector.detect_multimodal_inconsistency(video_path)

    print("\n" + "-"*70)
    print("RÉSULTAT DÉTECTION DEEPFAKE")
    print("-"*70)

    print(f"\n✓ Verdict: {result.get('verdict', 'N/A')}")
    print(f"✓ Score deepfake: {result.get('deepfake_score', 0):.1%}")
    print(f"✓ Confiance: {result.get('confidence', 0):.1%}")

    # Détails de l'analyse
    analysis = result.get('analysis', {})
    indicators = analysis.get('indicators', {})

    if indicators:
        print("\n📊 Indicateurs:")
        print(f"  • Audio deepfake: {indicators.get('audio_deepfake_prob', 0):.1%}")
        print(f"  • Vidéo deepfake: {indicators.get('video_deepfake_prob', 0):.1%}")
        print(f"  • Anomalie lip-sync: {indicators.get('lip_sync_anomaly', 0):.1%}")


def example_4_workflow_visualization():
    """
    Exemple 4: Visualisation du workflow.
    """
    print("\n" + "="*70)
    print("EXEMPLE 4: Visualisation du Workflow")
    print("="*70)

    config = load_config()
    fact_checker = MultiAgentFactChecker(
        llm_client=None,
        vector_store=None,
        config=config.get('agents', {})
    )

    # Affichage du workflow
    workflow_diagram = fact_checker.get_workflow_visualization()
    print(workflow_diagram)


def example_5_detailed_analysis():
    """
    Exemple 5: Analyse détaillée avec tous les agents.
    """
    print("\n" + "="*70)
    print("EXEMPLE 5: Analyse Détaillée Complète")
    print("="*70)

    config = load_config()
    fact_checker = MultiAgentFactChecker(
        llm_client=None,
        vector_store=None,
        config=config.get('agents', {})
    )

    claim = """
    Jean Dupont, PDG de TechCorp depuis 2020, a déclaré que les ventes
    de l'entreprise ont augmenté de 150% en 2024, faisant de TechCorp
    le leader mondial du secteur.
    """

    print(f"\n📋 Affirmation:\n{claim.strip()}")
    print("\n🔍 Analyse en cours...")

    result = fact_checker.check_claim(claim)

    # Affichage détaillé
    print("\n" + "-"*70)
    print("ANALYSE DÉTAILLÉE")
    print("-"*70)

    # 1. Classification
    claim_info = result.get('claim', {})
    print("\n1️⃣  CLASSIFICATION")
    print(f"   Thème: {claim_info.get('theme', 'N/A')}")
    print(f"   Complexité: {claim_info.get('complexity', 0)}/10")
    print(f"   Urgence: {claim_info.get('urgency', 0)}/10")

    # 2. Décomposition
    assertions = claim_info.get('decomposed_assertions', [])
    print(f"\n2️⃣  DÉCOMPOSITION ({len(assertions)} assertions)")
    for i, assertion in enumerate(assertions, 1):
        print(f"   {i}. {assertion}")

    # 3. Détection d'anomalies
    analysis = result.get('analysis', {})
    anomaly = analysis.get('anomaly_detection', {})
    print(f"\n3️⃣  DÉTECTION D'ANOMALIES")
    print(f"   Score moyen: {anomaly.get('average_score', 0):.2f}")
    print(f"   Assertions à risque: {anomaly.get('high_risk_assertions', 0)}")

    # 4. Preuves
    evidence = analysis.get('evidence_summary', {})
    print(f"\n4️⃣  PREUVES CONSULTÉES")
    print(f"   Total: {evidence.get('total_evidence', 0)}")
    print(f"   Crédibilité moyenne: {evidence.get('average_credibility', 0):.1%}")
    sources = evidence.get('sources_consulted', [])
    if sources:
        print(f"   Sources: {', '.join(sources[:5])}")

    # 5. Vérification
    verification = analysis.get('verification_results', {})
    breakdown = verification.get('verdict_breakdown', {})
    print(f"\n5️⃣  VÉRIFICATION")
    print(f"   Supportées: {breakdown.get('SUPPORTED', 0)}")
    print(f"   Réfutées: {breakdown.get('REFUTED', 0)}")
    print(f"   Insuffisantes: {breakdown.get('INSUFFICIENT_INFO', 0)}")

    # 6. Verdict final
    verdict = result.get('verdict', {})
    print(f"\n6️⃣  VERDICT FINAL")
    print(f"   {verdict.get('verdict_label', 'N/A')}")
    print(f"   Confiance: {verdict.get('confidence', 0):.1%}")

    # 7. Alertes
    alert = result.get('alert', {})
    if alert.get('should_alert'):
        print(f"\n⚠️  ALERTE: Niveau {alert.get('alert_level', 'N/A')}")
        for reason in alert.get('alert_reason', []):
            print(f"   • {reason}")

    # 8. Traçabilité
    traceability = result.get('traceability', {})
    trace = traceability.get('reasoning_trace', [])
    if trace:
        print(f"\n📝 TRACE DE RAISONNEMENT")
        for step in trace:
            print(f"   • {step}")


def main():
    """
    Fonction principale - exécute tous les exemples.
    """
    print("\n" + "="*70)
    print("   SYSTÈME MULTI-AGENTS DE DÉTECTION DE DÉSINFORMATION")
    print("   Phase 2 - Démonstration")
    print("="*70)

    # Menu
    examples = [
        ("Vérification simple", example_1_simple_claim),
        ("Vérifications multiples", example_2_multiple_claims),
        ("Détection deepfake", example_3_deepfake_detection),
        ("Visualisation workflow", example_4_workflow_visualization),
        ("Analyse détaillée", example_5_detailed_analysis)
    ]

    print("\n📚 Exemples disponibles:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"   {i}. {name}")
    print("   0. Tous les exemples")

    try:
        choice = input("\n👉 Choisissez un exemple (0-5): ").strip()

        if choice == '0':
            # Exécuter tous les exemples
            for name, func in examples:
                func()
        elif choice in ['1', '2', '3', '4', '5']:
            # Exécuter l'exemple choisi
            examples[int(choice) - 1][1]()
        else:
            print("❌ Choix invalide")
            return

    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du programme")
        return
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("✅ Démonstration terminée")
    print("="*70)
    print("\n📚 Pour plus d'informations:")
    print("   • Voir README.md")
    print("   • Voir projet-multi-agents-desinformation.md")
    print("   • Voir technique-approfondi.md")
    print()


if __name__ == "__main__":
    main()
