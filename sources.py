"""The research corpus every lab in this chapter works against.

Why a fixed corpus instead of a live web search:

1. **Reproducibility.** The chapter prints token counts, message counts, and
   offload thresholds as evidence. Those numbers only mean something if you can
   reproduce them, and a live search API returns different results every run.
2. **No second API key.** The only credential any lab needs is a model key.
3. **A controlled large document.** `nexus-earnings-call` is deliberately far
   larger than the harness's ~20,000-token offload threshold, so the filesystem
   lab can demonstrate automatic content offloading on demand rather than by
   waiting for a lucky search result.

Every company and document here is fictional. The market shape is realistic,
but inventing the vendors means the chapter cannot misquote a real one or go
stale when a real one pivots.
"""

from dataclasses import dataclass

from langchain_core.tools import tool


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    outlet: str
    date: str
    body: str


# --- The corpus -------------------------------------------------------------

_VEKTRA_LAUNCH = """\
Vektra has moved its managed vector search product to a serverless billing
model, decoupling storage from query capacity. Under the previous model
customers provisioned index replicas by the hour whether or not they were
serving traffic; under the new one they are billed for stored vectors and for
queries executed, with no idle floor.

The company says the change was driven by usage patterns in its own telemetry.
A majority of indexes it hosts serve fewer than ten queries per second at
median, with short bursts an order of magnitude higher. Provisioned capacity
forced those customers to pay for the burst continuously. Vektra's own figure
is that a typical retrieval-augmented generation workload spent between 60 and
80 percent of its bill on idle replicas.

The trade-off is cold-start latency. Vektra's documentation quotes a first-query
latency of 400 to 900 milliseconds against an index that has not been queried
recently, against 15 to 40 milliseconds warm. For interactive chat products that
is a visible pause. Vektra offers a paid warming tier that pins an index in
memory, which restores the old cost profile for latency-sensitive workloads.

Pricing is 0.32 US dollars per million stored vector-dimensions per month and
2.10 dollars per million queries. Vektra positions this as roughly a 45 percent
reduction for bursty workloads and acknowledges it is more expensive for
sustained high-throughput ones above approximately 400 queries per second.
"""

_ORBIT_FUNDING = """\
Orbit Index has raised 90 million dollars in a Series C round. The company says
the capital will fund a hybrid search engine combining dense vector retrieval
with classical sparse keyword scoring in a single query planner, rather than
running the two systems separately and fusing their results in application code.

Orbit's argument is that reciprocal rank fusion, the common approach to blending
dense and sparse results, is applied too late. Because it operates on two
already-truncated result lists, a document that ranks 51st on both retrievers is
discarded even when its combined evidence would place it in the top ten. Orbit
claims a unified planner that scores candidates once against both signals
improves recall at k=10 by 8 to 14 points on internal benchmarks.

The company reports annual recurring revenue "in the low eight figures" and
roughly 340 paying customers, which implies an average contract value well under
what its enterprise positioning would suggest. Analysts reading the round note
that the valuation, reported at 780 million dollars, prices in substantial
growth against a product that is not yet generally available.

Orbit's hybrid engine is in private preview with general availability guided to
the second half of next year.
"""

_OPENVEC_ADOPTION = """\
The most consequential development in vector search this year may be that fewer
teams are buying it. openvec, the open-source extension that adds vector indexing
to a conventional relational database, reports a threefold increase in downloads
year over year and now ships by default in three major managed database offerings.

The technical case for openvec is narrower than its advocates suggest, and
stronger than its detractors allow. Its index build times are materially worse
than a purpose-built engine — roughly 4 to 7 times slower on collections above
ten million vectors — and it does not support distributed index sharding, which
caps a single collection at what one machine can hold. Below that ceiling, recall
and query latency are competitive.

What openvec offers instead is the elimination of a synchronization problem.
Teams running a separate vector store must keep it consistent with the system of
record: every insert, update, and delete has to be mirrored, and every failure of
that mirroring produces a retrieval result referencing a row that no longer
exists. Practitioners describe this dual-write burden as the dominant operational
cost of a standalone vector database, exceeding both its licence fee and its
infrastructure bill.

The practical boundary appears to fall around 50 million vectors and 500 queries
per second. Below it, teams increasingly default to openvec. Above it, the
dedicated engines retain a clear advantage.
"""

_BENCHMARK_REPORT = """\
An independent benchmark of four vector search engines was published this
quarter, covering Vektra, Orbit Index, Nexus Data, and openvec across three
collection sizes and two recall targets.

At 1 million vectors and a 0.95 recall target, all four systems returned p95
query latencies within a 12-millisecond band, which the authors describe as
"effectively indistinguishable for application purposes." Differentiation
appeared only at 100 million vectors, where openvec was excluded because it
cannot shard a single collection across machines, and where Nexus Data's p95
latency degraded 3.2 times against its 1-million baseline while Vektra's
degraded 1.4 times.

The authors are explicit about two limitations that undercut vendor use of these
numbers. First, all tests used a single synthetic embedding distribution;
real corpora cluster differently and index structures are sensitive to that.
Second, every system was tuned by the benchmark authors rather than by its
vendor, and the authors concede they are likely to have under-tuned at least one.

Their stated conclusion is that engine choice is not the dominant variable in
retrieval quality for most applications, and that teams over-index on latency
benchmarks relative to the embedding model and chunking strategy, which the
authors argue account for substantially more of the observed variance in
end-to-end answer quality.
"""

_ANALYST_FORECAST = """\
A market forecast published this quarter sizes vector search infrastructure at
2.4 billion dollars in current annual spend, growing to a projected 11.9 billion
within five years, a compound annual growth rate of approximately 38 percent.

The methodology deserves scrutiny. The 2.4 billion figure counts spend on
managed vector database services, self-hosted licences, and an allocated share
of general-purpose database spend attributed to vector workloads. That third
category is roughly 40 percent of the total and is estimated rather than
observed; the analysts derive it from a survey of 210 engineering leaders asked
what proportion of their database spend supports retrieval.

The forecast's growth assumption is that retrieval-augmented generation moves
from pilot to production across the enterprise segment on a three-year horizon.
The same report notes that 62 percent of surveyed organizations describe their
retrieval deployments as "pilot or limited production," a figure essentially
unchanged from the prior year — which is difficult to reconcile with the growth
curve the forecast projects.

A more defensible reading of the same data is that spend is growing quickly from
a small base, that the addressable market is real, and that the five-year figure
embeds an adoption assumption the report's own survey does not support.
"""

_PRACTITIONER_SURVEY = """\
A survey of 1,180 engineers who have shipped a retrieval system to production
asked what they would change about their architecture given the chance.

The most common answer, at 44 percent, was to spend more effort on chunking and
document preparation and less on engine selection. The second, at 31 percent,
was to have started with the database they already operated rather than adopting
a specialized store. Only 9 percent said they would choose a different vector
engine, and 6 percent said they would move from a specialized engine to a
general-purpose one.

On operational pain, respondents ranked keeping the vector store synchronized
with the source of truth as the top burden (cited by 58 percent), ahead of query
latency (22 percent), index build time (19 percent), and cost (17 percent).

Asked what would most improve their retrieval quality, respondents named a
better embedding model (39 percent), better chunking (28 percent), hybrid
keyword-plus-vector search (21 percent), and a faster engine (4 percent).

The survey's own caveat is that it recruited through developer communities and
therefore over-represents smaller teams; only 14 percent of respondents worked
at organizations above 5,000 employees.
"""

# The oversized document. A quarterly earnings call transcript, expanded to
# comfortably exceed the harness's ~20,000-token offload threshold.
#
# The expansion is mechanical and this file says so plainly: the Q&A block below
# repeats with varying analyst names. That is enough to exercise the offload
# path honestly. Nothing in the chapter's argument depends on the filler being
# interesting -- only on the document being genuinely too large to sit in a
# context window, which is exactly the situation the harness handles for you.
_NEXUS_CALL_OPENING = """\
Nexus Data, Q3 earnings call. Prepared remarks followed by analyst Q&A.

CHIEF EXECUTIVE: Revenue for the quarter was 47.2 million dollars, up 61 percent
year over year. Net revenue retention was 118 percent, down from 131 percent in
the prior year. Gross margin improved to 71 percent from 64 percent, driven
primarily by the storage tiering work we shipped in the second quarter.

I want to address the retention number directly, because it is the figure that
matters most in this business and it moved in the wrong direction. Expansion
within our largest cohort remains strong. The compression is concentrated in
customers below 50,000 dollars of annual contract value, where we are seeing
consolidation onto general-purpose databases for smaller collections. We do not
think this segment is defensible on features, and we are not going to price to
defend it.

CHIEF FINANCIAL OFFICER: Operating expenses were 38.1 million, of which sales
and marketing was 19.4 million. Free cash flow was negative 6.2 million,
improving from negative 14.8 million a year ago. We ended the quarter with 210
million in cash and expect to reach breakeven on a free cash flow basis within
six quarters without additional financing.
"""

_NEXUS_CALL_QA_TEMPLATE = """\
ANALYST ({name}, {firm}): Thanks for taking the question. Can you talk about
the competitive dynamic at the low end, and whether the retention compression
you described is pricing or product?

CHIEF EXECUTIVE: It is neither, really. It is scope. A team with four million
vectors and modest query volume does not need a distributed engine, and the
extension in their existing database is now good enough at that size. When they
consolidate they are not switching vendors, they are removing a component. We
would rather concede that and put engineering into the collection sizes where
the architecture actually earns its cost.

ANALYST ({name}, {firm}): And on gross margin, is 71 percent the run rate or is
there more room from the tiering work?

CHIEF FINANCIAL OFFICER: There is more room. Tiering was rolled out to about
60 percent of eligible collections in the quarter. Fully deployed we think it
supports mid-to-high seventies, though the mix shift toward larger customers
works slightly against that because they negotiate harder.

ANALYST ({name}, {firm}): One more on the benchmark results published this
quarter, where your latency degradation at 100 million vectors was noticeably
worse than a competitor's. How do you respond to that?

CHIEF EXECUTIVE: The benchmark authors tuned every system themselves and said
plainly that they probably under-tuned at least one. We think we were that one.
That said, I am not going to stand here and claim the number is meaningless. Our
p95 at very large collection sizes is a real area of investment and we have work
shipping in the next two quarters. I would rather say that than argue with a
methodology that was, on balance, more transparent than most.
"""

_QA_PARTICIPANTS = [
    ("R. Almeida", "Kestrel Research"),
    ("J. Whitfield", "Northgate Securities"),
    ("M. Okonkwo", "Bellwether Capital"),
    ("S. Lindqvist", "Arbor Partners"),
    ("D. Fenwick", "Copperline Advisors"),
    ("T. Nakamura", "Vantage Point Equity"),
    ("A. Dubois", "Rivermark Analytics"),
    ("K. Sørensen", "Halden Group"),
    ("P. Venkataraman", "Cobalt Ridge"),
    ("L. Brennan", "Stonefield Research"),
    ("C. Achterberg", "Meridian Yield"),
    ("N. Petrova", "Larkspur Capital"),
    ("H. Castellanos", "Foxglove Partners"),
    ("E. Thorne", "Ashgrove Securities"),
    ("W. Mbeki", "Tidewater Research"),
    ("G. Rosenthal", "Pinnacle Crossing"),
    ("B. Ferreira", "Juniper Lane"),
    ("V. Kaur", "Blackthorn Equity"),
    ("O. Lindgren", "Westvale Research"),
    ("F. Moreau", "Camberwell Group"),
    ("I. Sokolov", "Greyshore Capital"),
    ("Y. Tanaka", "Elmridge Partners"),
    ("Q. Adeyemi", "Sandpiper Equity"),
    ("Z. Haddad", "Marlowe Research"),
    ("U. Novak", "Thistledown Capital"),
    ("X. Reyes", "Kingsbridge Advisors"),
    ("R. Kowalski", "Fernhill Securities"),
    ("J. Osei", "Brackenridge Group"),
    ("M. Lindholm", "Baldwin Reach"),
    ("S. Chaudhry", "Oakmere Analytics"),
    ("D. Ivanova", "Ridgeline Yield"),
    ("T. Bergström", "Hollowway Capital"),
    ("A. Nkemdirim", "Saltmarsh Partners",),
    ("K. Duarte", "Windmere Research"),
    ("P. Ó Súilleabháin", "Carrowmore Equity"),
    ("L. Yamamoto", "Aldergate Capital"),
    ("C. Vasquez", "Netherfield Group"),
    ("N. Björk", "Quarrystone Advisors"),
    ("H. Mensah", "Larchfield Securities"),
    ("E. Kovačević", "Thornbury Research"),
    ("W. Pereira", "Highmoor Capital"),
    ("G. Andersson", "Fairweather Partners"),
    ("B. Nasser", "Underhill Equity"),
    ("V. Solberg", "Grantham Yield"),
    ("O. Machado", "Sedgewick Research"),
    ("F. Ibrahim", "Ravensworth Group"),
    ("I. Laurent", "Coldharbour Capital"),
    ("Y. Petrov", "Millbrook Advisors"),
    ("Q. Fitzgerald", "Ashcombe Partners"),
    ("Z. Nilsson", "Bramblewood Equity",),
    ("U. Delgado", "Ferngrove Research"),
    ("X. Aliyev", "Stonecross Capital"),
]


def _build_nexus_transcript() -> str:
    """Assemble the oversized transcript from a repeating Q&A block."""
    blocks = [_NEXUS_CALL_OPENING]
    for name, firm in _QA_PARTICIPANTS:
        blocks.append(_NEXUS_CALL_QA_TEMPLATE.format(name=name, firm=firm))
    blocks.append(
        "CHIEF EXECUTIVE: That is all the time we have. Thank you all for "
        "joining, and we will speak again next quarter.\n"
    )
    return "\n".join(blocks)


SOURCES: dict[str, Source] = {
    s.id: s
    for s in [
        Source(
            id="vektra-serverless",
            title="Vektra moves managed vector search to serverless billing",
            outlet="Vektra engineering blog",
            date="2026-02-11",
            body=_VEKTRA_LAUNCH,
        ),
        Source(
            id="orbit-series-c",
            title="Orbit Index raises $90M Series C for hybrid retrieval",
            outlet="The Ledger",
            date="2026-03-04",
            body=_ORBIT_FUNDING,
        ),
        Source(
            id="openvec-adoption",
            title="The quiet consolidation onto general-purpose databases",
            outlet="Retrieval Weekly",
            date="2026-01-28",
            body=_OPENVEC_ADOPTION,
        ),
        Source(
            id="engine-benchmark",
            title="Independent benchmark: four vector engines, three scales",
            outlet="Open Benchmark Collective",
            date="2026-03-19",
            body=_BENCHMARK_REPORT,
        ),
        Source(
            id="analyst-forecast",
            title="Vector search infrastructure: $2.4B today, $11.9B by 2031",
            outlet="Halden Group",
            date="2026-02-25",
            body=_ANALYST_FORECAST,
        ),
        Source(
            id="practitioner-survey",
            title="What 1,180 engineers would change about their retrieval stack",
            outlet="Retrieval Weekly",
            date="2026-03-11",
            body=_PRACTITIONER_SURVEY,
        ),
        Source(
            id="nexus-earnings-call",
            title="Nexus Data Q3 earnings call — full transcript",
            outlet="Nexus Data investor relations",
            date="2026-03-22",
            body=_build_nexus_transcript(),
        ),
    ]
}


# --- Token accounting -------------------------------------------------------


def approx_tokens(text: str) -> int:
    """A cheap, stable token estimate: roughly four characters per token.

    Deliberately not a real tokenizer. Every number the chapter prints uses this
    same estimate, so the comparisons between them are consistent, which is all
    the argument needs. Where an exact count matters the labs read the provider's
    reported usage instead.
    """
    return len(text) // 4


# --- The tools the agents actually call -------------------------------------


@tool
def search_sources(query: str) -> str:
    """Search the research corpus for sources relevant to a query.

    Returns a compact index of matching sources: id, title, outlet, date, and
    approximate size. Use fetch_source to read one in full.
    """
    terms = [t for t in query.lower().split() if len(t) > 3]
    scored: list[tuple[int, Source]] = []
    for src in SOURCES.values():
        haystack = f"{src.title} {src.outlet} {src.body}".lower()
        score = sum(haystack.count(term) for term in terms)
        scored.append((score, src))

    # Always return the whole index; ranking just orders it. A research agent
    # should be able to see what exists, not only what matched a keyword.
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))

    lines = [f"{len(scored)} sources in corpus (ranked for: {query!r})", ""]
    for score, src in scored:
        lines.append(
            f"- id={src.id} | {src.title}\n"
            f"    {src.outlet}, {src.date} | ~{approx_tokens(src.body):,} tokens"
            f" | relevance {score}"
        )
    return "\n".join(lines)


@tool
def fetch_source(source_id: str) -> str:
    """Fetch the full text of one source by its id.

    Note that sources vary enormously in size. Check the size reported by
    search_sources before fetching: one source in this corpus is large enough
    to dominate a context window on its own.
    """
    src = SOURCES.get(source_id)
    if src is None:
        available = ", ".join(sorted(SOURCES))
        return f"No source with id {source_id!r}. Available ids: {available}"
    return (
        f"# {src.title}\n"
        f"Source: {src.outlet} | {src.date} | id={src.id}\n\n"
        f"{src.body}"
    )


RESEARCH_TOOLS = [search_sources, fetch_source]


# --- The task every lab works on --------------------------------------------

BRIEF_TASK = (
    "Produce a competitive brief on the vector search infrastructure market "
    "for our head of engineering, who is deciding whether to adopt a "
    "specialized vector database or extend the database we already run.\n\n"
    "The brief must have exactly four sections:\n"
    "1. Market shape — size, growth, and how much confidence the evidence supports\n"
    "2. The vendors — who is positioned how, and on what basis\n"
    "3. The case against adopting a specialized engine\n"
    "4. A recommendation, with the condition that would change it\n\n"
    "Ground every claim in the corpus. Where sources disagree or a number is "
    "weakly supported, say so rather than averaging them."
)


if __name__ == "__main__":
    print(f"{len(SOURCES)} sources in the corpus:\n")
    total = 0
    for src in SOURCES.values():
        n = approx_tokens(src.body)
        total += n
        print(f"  {src.id:<22} ~{n:>7,} tokens  {src.title[:44]}")
    print(f"\n  {'TOTAL':<22} ~{total:>7,} tokens")
    biggest = max(SOURCES.values(), key=lambda s: len(s.body))
    print(
        f"\nThe six short sources are small enough to read freely. "
        f"{biggest.id!r},\nat ~{approx_tokens(biggest.body):,} tokens, is not: it exceeds the "
        "harness's ~20,000-token\noffload threshold on its own. That asymmetry is what the "
        "filesystem\nlab is built to demonstrate."
    )
