"""Agent 1: Collector and Indexer - Monitors and collects content from various sources."""

from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.types import FactCheckingState, Source, Evidence
from utils.logger import log
from utils.helpers import extract_domain, extract_urls
from utils.credibility import score_source_credibility
from config.settings import settings


class CollectorAgent(BaseAgent):
    """
    Collector and Indexer Agent.

    Responsibilities:
    - Monitor public sources (Twitter, Reddit, News APIs, etc.)
    - Collect and normalize data
    - Index sources with metadata
    - Detect potentially viral or anomalous content
    """

    def __init__(self):
        super().__init__(
            name="CollectorAgent",
            description="Monitors and collects content from various public sources"
        )
        self.sources_index: Dict[str, Source] = {}

    async def process(self, state: FactCheckingState) -> FactCheckingState:
        """
        Process claim and collect related evidence from various sources.

        Args:
            state: Current fact-checking state

        Returns:
            Updated state with collected evidence
        """
        self.log_action(state, "Starting evidence collection")

        # Extract URLs from claim if any
        urls = extract_urls(state.original_claim)

        # Collect evidence from multiple sources in parallel
        collection_tasks = [
            self._collect_from_web_search(state.original_claim),
            self._collect_from_news_apis(state.original_claim),
            self._collect_from_social_media(state.original_claim),
        ]

        if urls:
            collection_tasks.append(self._analyze_claim_urls(urls))

        # Execute all collection tasks in parallel
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)

        # Aggregate all evidence
        all_evidence = []
        for result in results:
            if isinstance(result, list):
                all_evidence.extend(result)
            elif isinstance(result, Exception):
                log.error(f"Collection task failed: {str(result)}")

        # Update state
        state.evidence_retrieved.extend(all_evidence)

        self.log_action(
            state,
            f"Collected {len(all_evidence)} pieces of evidence from {len(set(e.source.domain for e in all_evidence))} unique sources"
        )

        return state

    async def _collect_from_web_search(self, query: str) -> List[Evidence]:
        """
        Collect evidence from web search (would use Google Custom Search API or similar).

        Args:
            query: Search query

        Returns:
            List of evidence pieces
        """
        evidence_list = []

        # TODO: Implement actual web search API integration
        # For now, this is a placeholder showing the expected structure
        log.info(f"[{self.name}] Searching web for: {query}")

        # Placeholder - would call actual search API
        # Example structure:
        # results = await search_api.search(query, num_results=10)
        # for result in results:
        #     evidence = self._create_evidence_from_result(result, query)
        #     evidence_list.append(evidence)

        return evidence_list

    async def _collect_from_news_apis(self, query: str) -> List[Evidence]:
        """
        Collect evidence from news APIs.

        Args:
            query: Search query

        Returns:
            List of evidence pieces
        """
        evidence_list = []

        if not settings.news_api_key:
            log.warning(f"[{self.name}] News API key not configured, skipping news collection")
            return evidence_list

        log.info(f"[{self.name}] Searching news APIs for: {query}")

        # TODO: Implement NewsAPI integration
        # from newsapi import NewsApiClient
        # newsapi = NewsApiClient(api_key=settings.news_api_key)
        # articles = newsapi.get_everything(q=query, language='fr', sort_by='relevancy', page_size=10)
        #
        # for article in articles.get('articles', []):
        #     source = Source(
        #         url=article['url'],
        #         domain=extract_domain(article['url']),
        #         title=article.get('title'),
        #         credibility_score=score_source_credibility(article['url']),
        #         timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
        #     )
        #     evidence = Evidence(
        #         text=article.get('description', '') + ' ' + article.get('content', '')[:500],
        #         source=source,
        #         relevance_score=0.7  # Would be computed by semantic similarity
        #     )
        #     evidence_list.append(evidence)

        return evidence_list

    async def _collect_from_social_media(self, query: str) -> List[Evidence]:
        """
        Collect evidence from social media platforms (Twitter, Reddit).

        Args:
            query: Search query

        Returns:
            List of evidence pieces
        """
        evidence_list = []

        log.info(f"[{self.name}] Searching social media for: {query}")

        # Twitter/X collection
        if settings.twitter_bearer_token:
            twitter_evidence = await self._collect_from_twitter(query)
            evidence_list.extend(twitter_evidence)

        # Reddit collection
        if settings.reddit_client_id and settings.reddit_client_secret:
            reddit_evidence = await self._collect_from_reddit(query)
            evidence_list.extend(reddit_evidence)

        return evidence_list

    async def _collect_from_twitter(self, query: str) -> List[Evidence]:
        """
        Collect evidence from Twitter/X.

        Args:
            query: Search query

        Returns:
            List of evidence pieces
        """
        evidence_list = []

        # TODO: Implement Twitter API v2 integration
        # import tweepy
        # client = tweepy.Client(bearer_token=settings.twitter_bearer_token)
        # tweets = client.search_recent_tweets(
        #     query=query,
        #     max_results=10,
        #     tweet_fields=['created_at', 'author_id', 'public_metrics']
        # )
        #
        # for tweet in tweets.data or []:
        #     source = Source(
        #         url=f"https://twitter.com/i/web/status/{tweet.id}",
        #         domain="twitter.com",
        #         credibility_score=0.3,  # Social media has lower base credibility
        #         timestamp=tweet.created_at
        #     )
        #     evidence = Evidence(
        #         text=tweet.text,
        #         source=source,
        #         relevance_score=0.5
        #     )
        #     evidence_list.append(evidence)

        return evidence_list

    async def _collect_from_reddit(self, query: str) -> List[Evidence]:
        """
        Collect evidence from Reddit.

        Args:
            query: Search query

        Returns:
            List of evidence pieces
        """
        evidence_list = []

        # TODO: Implement Reddit API integration
        # import praw
        # reddit = praw.Reddit(
        #     client_id=settings.reddit_client_id,
        #     client_secret=settings.reddit_client_secret,
        #     user_agent=settings.reddit_user_agent
        # )
        #
        # for submission in reddit.subreddit('all').search(query, limit=10):
        #     source = Source(
        #         url=submission.url,
        #         domain="reddit.com",
        #         title=submission.title,
        #         credibility_score=0.35,
        #         timestamp=datetime.fromtimestamp(submission.created_utc)
        #     )
        #     evidence = Evidence(
        #         text=f"{submission.title}\n{submission.selftext[:500]}",
        #         source=source,
        #         relevance_score=0.5
        #     )
        #     evidence_list.append(evidence)

        return evidence_list

    async def _analyze_claim_urls(self, urls: List[str]) -> List[Evidence]:
        """
        Analyze URLs mentioned in the claim.

        Args:
            urls: List of URLs to analyze

        Returns:
            List of evidence pieces
        """
        evidence_list = []

        for url in urls[:5]:  # Limit to first 5 URLs
            try:
                domain = extract_domain(url)
                credibility = score_source_credibility(url, domain)

                source = Source(
                    url=url,
                    domain=domain,
                    credibility_score=credibility,
                    timestamp=datetime.now()
                )

                # TODO: Fetch and extract content from URL
                # import aiohttp
                # async with aiohttp.ClientSession() as session:
                #     async with session.get(url) as response:
                #         html = await response.text()
                #         # Extract main content using BeautifulSoup
                #         text = extract_main_content(html)

                evidence = Evidence(
                    text=f"URL referenced in claim: {url}",
                    source=source,
                    relevance_score=0.8  # URLs in claim are highly relevant
                )

                evidence_list.append(evidence)
                self._index_source(source)

            except Exception as e:
                log.error(f"[{self.name}] Error analyzing URL {url}: {str(e)}")

        return evidence_list

    def _index_source(self, source: Source) -> None:
        """
        Add source to internal index for tracking.

        Args:
            source: Source to index
        """
        self.sources_index[source.url] = source

    def _create_evidence_from_result(
        self,
        result: Dict,
        query: str,
        relevance_score: Optional[float] = None
    ) -> Evidence:
        """
        Create Evidence object from search result.

        Args:
            result: Search result dictionary
            query: Original query
            relevance_score: Optional relevance score

        Returns:
            Evidence object
        """
        url = result.get('url', '')
        domain = extract_domain(url)

        source = Source(
            url=url,
            domain=domain,
            title=result.get('title'),
            credibility_score=score_source_credibility(url, domain),
            timestamp=result.get('timestamp', datetime.now())
        )

        evidence = Evidence(
            text=result.get('snippet', ''),
            source=source,
            relevance_score=relevance_score or 0.5
        )

        self._index_source(source)

        return evidence
