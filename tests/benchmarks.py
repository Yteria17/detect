"""Benchmarking suite for the multi-agent fact-checking system.

This module provides comprehensive benchmarking and evaluation metrics for
assessing the performance, accuracy, and efficiency of the system.
"""

import time
import json
import asyncio
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""

    test_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_latency_seconds: float
    median_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    throughput_per_second: float
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def __str__(self) -> str:
        """String representation."""
        return f"""
Benchmark: {self.test_name}
{'=' * 60}
Accuracy Metrics:
  - Accuracy:  {self.accuracy:.2%}
  - Precision: {self.precision:.2%}
  - Recall:    {self.recall:.2%}
  - F1 Score:  {self.f1_score:.2%}

Performance Metrics:
  - Avg Latency:    {self.avg_latency_seconds:.2f}s
  - Median Latency: {self.median_latency_seconds:.2f}s
  - P95 Latency:    {self.p95_latency_seconds:.2f}s
  - P99 Latency:    {self.p99_latency_seconds:.2f}s
  - Throughput:     {self.throughput_per_second:.2f} req/s

Confusion Matrix:
  - True Positives:  {self.true_positives}
  - False Positives: {self.false_positives}
  - True Negatives:  {self.true_negatives}
  - False Negatives: {self.false_negatives}
  - Total Samples:   {self.total_samples}
"""


class FactCheckBenchmark:
    """Benchmark suite for fact-checking system."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []

    def calculate_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        latencies: List[float],
        test_name: str = "Unnamed Test",
    ) -> BenchmarkResult:
        """Calculate comprehensive metrics from predictions.

        Args:
            predictions: List of predicted verdicts
            ground_truth: List of true verdicts
            latencies: List of processing times in seconds
            test_name: Name of the benchmark test

        Returns:
            BenchmarkResult with all calculated metrics
        """
        assert len(predictions) == len(ground_truth) == len(latencies)

        # Calculate confusion matrix
        # For simplicity, treating "REFUTED" as positive class
        tp = sum(
            1
            for pred, truth in zip(predictions, ground_truth)
            if pred == "REFUTED" and truth == "REFUTED"
        )
        fp = sum(
            1
            for pred, truth in zip(predictions, ground_truth)
            if pred == "REFUTED" and truth != "REFUTED"
        )
        tn = sum(
            1
            for pred, truth in zip(predictions, ground_truth)
            if pred != "REFUTED" and truth != "REFUTED"
        )
        fn = sum(
            1
            for pred, truth in zip(predictions, ground_truth)
            if pred != "REFUTED" and truth == "REFUTED"
        )

        # Calculate accuracy metrics
        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate latency metrics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)

        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p99_index = int(len(sorted_latencies) * 0.99)

        p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
        p99_latency = sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1]

        # Calculate throughput
        total_time = sum(latencies)
        throughput = total / total_time if total_time > 0 else 0

        result = BenchmarkResult(
            test_name=test_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            avg_latency_seconds=avg_latency,
            median_latency_seconds=median_latency,
            p95_latency_seconds=p95_latency,
            p99_latency_seconds=p99_latency,
            throughput_per_second=throughput,
            total_samples=total,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            timestamp=datetime.now(),
        )

        self.results.append(result)
        return result

    def run_baseline_comparison(
        self,
        system_predictions: List[Tuple[str, float]],  # (verdict, latency)
        baseline_predictions: List[Tuple[str, float]],
        ground_truth: List[str],
    ) -> Dict[str, BenchmarkResult]:
        """Compare system against baseline.

        Args:
            system_predictions: List of (verdict, latency) from our system
            baseline_predictions: List of (verdict, latency) from baseline
            ground_truth: True verdicts

        Returns:
            Dict with "system" and "baseline" BenchmarkResults
        """
        system_verdicts = [v for v, _ in system_predictions]
        system_latencies = [l for _, l in system_predictions]

        baseline_verdicts = [v for v, _ in baseline_predictions]
        baseline_latencies = [l for _, l in baseline_predictions]

        system_result = self.calculate_metrics(
            system_verdicts, ground_truth, system_latencies, "Multi-Agent System"
        )

        baseline_result = self.calculate_metrics(
            baseline_verdicts, ground_truth, baseline_latencies, "Baseline System"
        )

        return {"system": system_result, "baseline": baseline_result}

    def stress_test(
        self, fact_check_func, num_requests: int = 100, concurrency: int = 10
    ) -> Dict:
        """Run stress test with concurrent requests.

        Args:
            fact_check_func: Async function to call for fact-checking
            num_requests: Total number of requests to make
            concurrency: Number of concurrent requests

        Returns:
            Stress test results
        """
        import asyncio

        async def worker(request_id: int) -> Dict:
            """Worker to process a single request."""
            claim = f"Test claim number {request_id}"
            start_time = time.time()

            try:
                result = await fact_check_func(claim)
                latency = time.time() - start_time
                return {
                    "request_id": request_id,
                    "success": True,
                    "latency": latency,
                    "result": result,
                }
            except Exception as e:
                latency = time.time() - start_time
                return {
                    "request_id": request_id,
                    "success": False,
                    "latency": latency,
                    "error": str(e),
                }

        async def run_stress():
            """Run all requests with controlled concurrency."""
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_worker(request_id: int):
                async with semaphore:
                    return await worker(request_id)

            tasks = [bounded_worker(i) for i in range(num_requests)]
            return await asyncio.gather(*tasks)

        # Run the stress test
        start_time = time.time()
        results = asyncio.run(run_stress())
        total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        latencies = [r["latency"] for r in successful]

        return {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / num_requests,
            "total_time_seconds": total_time,
            "throughput_per_second": num_requests / total_time,
            "avg_latency_seconds": statistics.mean(latencies) if latencies else 0,
            "median_latency_seconds": statistics.median(latencies)
            if latencies
            else 0,
            "max_latency_seconds": max(latencies) if latencies else 0,
            "min_latency_seconds": min(latencies) if latencies else 0,
            "concurrency": concurrency,
        }

    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Generate comprehensive benchmark report.

        Args:
            output_file: Path to save the report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary(),
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nBenchmark report saved to: {output_file}")
        print("\nSummary:")
        print(self._generate_summary_text())

        return report

    def _generate_summary(self) -> Dict:
        """Generate summary statistics across all tests."""
        if not self.results:
            return {}

        return {
            "avg_accuracy": statistics.mean(r.accuracy for r in self.results),
            "avg_precision": statistics.mean(r.precision for r in self.results),
            "avg_recall": statistics.mean(r.recall for r in self.results),
            "avg_f1_score": statistics.mean(r.f1_score for r in self.results),
            "avg_latency": statistics.mean(r.avg_latency_seconds for r in self.results),
            "best_accuracy": max(r.accuracy for r in self.results),
            "best_f1_score": max(r.f1_score for r in self.results),
            "fastest_avg_latency": min(
                r.avg_latency_seconds for r in self.results
            ),
        }

    def _generate_summary_text(self) -> str:
        """Generate human-readable summary."""
        summary = self._generate_summary()

        return f"""
Overall Performance Summary
{'=' * 60}
Average Accuracy:  {summary.get('avg_accuracy', 0):.2%}
Average Precision: {summary.get('avg_precision', 0):.2%}
Average Recall:    {summary.get('avg_recall', 0):.2%}
Average F1 Score:  {summary.get('avg_f1_score', 0):.2%}

Best Accuracy:     {summary.get('best_accuracy', 0):.2%}
Best F1 Score:     {summary.get('best_f1_score', 0):.2%}

Average Latency:   {summary.get('avg_latency', 0):.2f}s
Fastest Latency:   {summary.get('fastest_avg_latency', 0):.2f}s
"""


# Example usage and test cases


def create_sample_dataset() -> Dict[str, List]:
    """Create a sample dataset for testing.

    Returns:
        Dict with claims, ground_truth, and metadata
    """
    return {
        "claims": [
            "The Earth is flat",  # REFUTED
            "Water boils at 100°C at sea level",  # SUPPORTED
            "COVID-19 vaccines contain microchips",  # REFUTED
            "The capital of France is Paris",  # SUPPORTED
            "5G towers cause COVID-19",  # REFUTED
            "The sun is a star",  # SUPPORTED
            "The moon landing was faked",  # REFUTED
            "Humans need oxygen to breathe",  # SUPPORTED
            "Vaccines cause autism",  # REFUTED
            "The Pacific Ocean is the largest ocean",  # SUPPORTED
        ],
        "ground_truth": [
            "REFUTED",
            "SUPPORTED",
            "REFUTED",
            "SUPPORTED",
            "REFUTED",
            "SUPPORTED",
            "REFUTED",
            "SUPPORTED",
            "REFUTED",
            "SUPPORTED",
        ],
    }


if __name__ == "__main__":
    # Example: Run benchmarks
    benchmark = FactCheckBenchmark()

    # Create sample data
    dataset = create_sample_dataset()

    # Simulate predictions (in production, these would come from the actual system)
    predictions = [
        "REFUTED",
        "SUPPORTED",
        "REFUTED",
        "SUPPORTED",
        "REFUTED",
        "SUPPORTED",
        "REFUTED",
        "SUPPORTED",
        "REFUTED",
        "SUPPORTED",
    ]

    # Simulate latencies (in production, these would be measured)
    latencies = [2.5, 1.8, 3.2, 1.5, 2.9, 1.7, 3.5, 1.6, 2.8, 1.9]

    # Calculate metrics
    result = benchmark.calculate_metrics(
        predictions, dataset["ground_truth"], latencies, "Sample Test"
    )

    print(result)

    # Generate report
    benchmark.generate_report("benchmark_results.json")
