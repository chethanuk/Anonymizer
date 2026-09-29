"""Detector returns 'Aria and Leo' as one first_name span (issue #142); run the real post-detection pipeline."""
from anonymizer.engine.detection.postprocess import (
    EntitySpan, apply_augmented_entities, build_tagged_text, expand_entity_occurrences, group_entities_by_value,
)

text = "Bobby Watford lives with his children, Aria and Leo. On weekends Aria coaches soccer."
detected = [
    EntitySpan("p", "Bobby Watford", "full_name", 0, 13, 0.95, "detector"),
    EntitySpan("c", "Aria and Leo", "first_name", text.index("Aria"), text.index("Leo") + 3, 0.9, "detector"),
]
merged = apply_augmented_entities(text=text, entities=detected, augmented_output={"entities": []})
final = expand_entity_occurrences(text, merged)
print("entities_by_value:", [g["value"] for g in group_entities_by_value(final)])
print("tagged:", build_tagged_text(text, final))
redacted = text
for e in sorted(final, key=lambda e: e.start_position, reverse=True):
    redacted = redacted[: e.start_position] + f"[{e.label.upper()}]" + redacted[e.end_position :]
print("redacted:", redacted)
print("LEAK: 'Aria' still in output" if "Aria" in redacted else "OK: no 'Aria' left in output")
