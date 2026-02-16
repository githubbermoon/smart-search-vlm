import unittest

from mm_stack.intent_reranker import rerank_with_intent
from mm_stack.query_planner import parse_query


class IntentRankingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "image_id": "old_man_meal",
                "caption": "an old man in white check shirt having meal",
                "summary": "elderly male dining at restaurant table",
                "tags": ["old", "man", "meal", "white", "checkered", "shirt"],
                "score": 0.82,
            },
            {
                "image_id": "elderly_couple",
                "caption": "an elderly couple having lunch",
                "summary": "senior man and woman dining together",
                "tags": ["elderly", "couple", "lunch", "restaurant"],
                "score": 0.84,
            },
            {
                "image_id": "car_bike_blue",
                "caption": "a blue car next to a bike",
                "summary": "car and bicycle parked side by side",
                "tags": ["car", "bike", "blue", "next to"],
                "relation_evidence": [{"relation": "next_to", "confidence": 0.95}],
                "score": 0.79,
            },
            {
                "image_id": "red_striped_person",
                "caption": "person wearing red striped shirt",
                "summary": "portrait with patterned clothing",
                "tags": ["person", "red", "striped", "shirt"],
                "score": 0.8,
            },
            {
                "image_id": "walking_dog_beach",
                "caption": "a person walking dog on beach",
                "summary": "coastline scene with dog and human",
                "tags": ["walking", "dog", "beach", "person"],
                "score": 0.81,
            },
            {
                "image_id": "holding_camera",
                "caption": "person holding camera near lake",
                "summary": "photographer with camera in hand",
                "tags": ["person", "holding", "camera"],
                "score": 0.83,
            },
            {
                "image_id": "david_hat",
                "caption": "David smiling while wearing a hat",
                "summary": "portrait of david in black hat",
                "tags": ["david", "hat", "person"],
                "mentions": [{"mention": "David", "mention_type": "name", "confidence": 0.9}],
                "score": 0.78,
            },
            {
                "image_id": "banana_clock_plant",
                "caption": "banana and clock next to plant on table",
                "summary": "objects arranged with banana, clock and plant",
                "tags": ["banana", "clock", "plant", "next to"],
                "relation_evidence": [{"relation": "next_to", "confidence": 0.8}],
                "score": 0.76,
            },
            {
                "image_id": "noise_colorize",
                "caption": "remote sensing sar colorization paper",
                "summary": "model predicts colorized outputs",
                "tags": ["colorization", "paper"],
                "score": 0.9,
            },
        ]

    def _rank(self, query: str):
        intent = parse_query(query)
        return rerank_with_intent(
            [dict(r) for r in self.rows],
            intent,
            retrieval_weight=0.6,
            attribute_weight=0.25,
            relation_weight=0.2,
            required_entity_penalty=0.35,
            activity_boost=0.12,
            color_boost=0.2,
            pattern_boost=0.2,
            presence_required=True,
        )

    def test_scenarios(self):
        expectations = {
            "old man": {"old_man_meal", "elderly_couple"},
            "white check shirt having meal": "old_man_meal",
            "car next to bike": "car_bike_blue",
            "car next to bike what color is the car": "car_bike_blue",
            "elderly couple lunch": "elderly_couple",
            "person wearing red striped shirt": "red_striped_person",
            "walking dog on beach": "walking_dog_beach",
            "person holding camera": "holding_camera",
            "David with hat": "david_hat",
            "objects containing banana and clock": "banana_clock_plant",
        }
        for query, expected in expectations.items():
            with self.subTest(query=query):
                ranked = self._rank(query)
                if isinstance(expected, set):
                    self.assertIn(ranked[0]["image_id"], expected)
                else:
                    self.assertEqual(ranked[0]["image_id"], expected)
                self.assertGreaterEqual(ranked[0]["final_score"], 0.4)

    def test_typo_tolerance(self):
        ranked = self._rank("jwellery reciept enviroment")
        self.assertTrue(len(ranked) > 0)


if __name__ == "__main__":
    unittest.main()
