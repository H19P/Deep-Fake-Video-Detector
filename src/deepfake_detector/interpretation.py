"""Plain-language explanations for provisional fake-class score bands."""

def explain_fake_score(score: float | None, low: float = .30, high: float = .70) -> tuple[str, str]:
    """Explain score direction without translating it into probability or proof."""
    if score is None:
        return (
            "Inconclusive",
            "There were too few usable face predictions to assess this video.",
        )
    if not 0 <= score <= 1:
        raise ValueError("fake-class score must be between 0 and 1")
    if score >= high:
        return (
            "High fake-class score",
            f"The model leans toward fake-class patterns (score {score:.2f}; high band starts at {high:.2f}). "
            "This is not a probability that the video is fake and does not prove manipulation.",
        )
    if score <= low:
        return (
            "Low fake-class score",
            f"The model leans toward real-class patterns (score {score:.2f}; low band ends at {low:.2f}). "
            "This does not prove the video is authentic.",
        )
    return (
        "Inconclusive",
        f"The model score ({score:.2f}) is between {low:.2f} and {high:.2f}, so this run does not lean clearly either way.",
    )
