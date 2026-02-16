import unittest

from mm_stack.query_planner import parse_query, parse_query_intent


class QueryPlannerTests(unittest.TestCase):
    def test_old_man(self):
        intent = parse_query("old man")
        self.assertIn("man", intent.tokens_normalized)
        self.assertIn("old", intent.attribute_terms)
        self.assertTrue(intent.require_person)

    def test_white_check_shirt_having_meal(self):
        intent = parse_query("white check shirt having meal")
        self.assertIn("shirt", intent.clothing_terms)
        self.assertIn("white", intent.appearance["colors"])
        self.assertTrue(any(x in intent.appearance["patterns"] for x in ["check", "checkered", "plaid"]))
        self.assertIn("meal", intent.activity_terms)
        self.assertTrue(intent.require_person)

    def test_car_next_to_bike_color(self):
        intent = parse_query("car next to bike what color is the car")
        self.assertIn("next_to", intent.relation_terms)
        self.assertIn("car", intent.retrieval_terms)
        self.assertIn("bike", intent.retrieval_terms)
        self.assertIn("car", intent.presence_terms)
        self.assertIn("bike", intent.presence_terms)

    def test_banana_and_clock_next_to_plant(self):
        intent = parse_query("banana and clock next to plant")
        self.assertIn("next_to", intent.relation_terms)
        self.assertIn("banana", intent.retrieval_terms)
        self.assertIn("clock", intent.retrieval_terms)
        self.assertTrue(intent.query_type_flags.compositional)

    def test_legacy_wrapper(self):
        legacy = parse_query_intent("white check shirt, having meal")
        self.assertTrue(legacy.requires_person)
        self.assertTrue(legacy.requires_clothing)
        self.assertIn("shirt", legacy.clothing_terms)


if __name__ == "__main__":
    unittest.main()
