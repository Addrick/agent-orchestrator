import unittest
from decimal import Decimal

from shop.cart import Cart


class CartTest(unittest.TestCase):
    def test_add_and_subtotal(self):
        cart = Cart()
        cart.add("apple", "0.50", 4)
        cart.add("pear", "1.25")
        self.assertEqual(cart.subtotal(), Decimal("3.25"))

    def test_adding_the_same_sku_keeps_the_first_price(self):
        cart = Cart()
        cart.add("apple", "0.50", 2)
        cart.add("apple", "9.99", 1)
        self.assertEqual(cart.items["apple"], (Decimal("0.50"), 3))

    def test_remove(self):
        cart = Cart()
        cart.add("apple", "0.50", 3)
        cart.remove("apple", 2)
        self.assertEqual(cart.items["apple"][1], 1)
        cart.remove("apple", 5)
        self.assertNotIn("apple", cart.items)

    def test_order_discount_rounds_half_up(self):
        cart = Cart()
        cart.add("widget", "0.35", 1)
        self.assertEqual(cart.total(10), Decimal("0.32"))

    def test_bulk_discount_at_exactly_ten_units(self):
        cart = Cart()
        cart.add("bolt", "2.00", 10)
        self.assertEqual(cart.line_total("bolt"), Decimal("18.00"))


if __name__ == "__main__":
    unittest.main()
