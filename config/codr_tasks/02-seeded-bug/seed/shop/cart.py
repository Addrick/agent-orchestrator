"""A small shopping cart: prices in Decimal, a bulk discount per line, and an
optional percentage discount on the whole order."""

from decimal import ROUND_HALF_UP, Decimal

# A line with this many units or more gets BULK_DISCOUNT off that line.
BULK_QTY = 10
BULK_DISCOUNT = Decimal("0.10")


class Cart:
    def __init__(self) -> None:
        self.items: dict = {}  # sku -> (unit price, quantity)

    def add(self, sku: str, unit_price, qty: int = 1) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        price = Decimal(str(unit_price))
        if sku in self.items:
            old_price, old_qty = self.items[sku]
            self.items[sku] = (old_price, old_qty + qty)
        else:
            self.items[sku] = (price, qty)

    def remove(self, sku: str, qty: int = 1) -> None:
        price, have = self.items[sku]
        if qty >= have:
            del self.items[sku]
        else:
            self.items[sku] = (price, have - qty)

    def line_total(self, sku: str) -> Decimal:
        price, qty = self.items[sku]
        total = price * qty
        if qty > BULK_QTY:
            total -= total * BULK_DISCOUNT
        return total

    def subtotal(self) -> Decimal:
        return sum((self.line_total(sku) for sku in self.items), Decimal("0"))

    def total(self, discount_percent=0) -> Decimal:
        """Subtotal minus a percentage discount, rounded to cents (half up)."""
        sub = self.subtotal()
        discounted = sub - sub * Decimal(str(discount_percent)) / 100
        return discounted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
