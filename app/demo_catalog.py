"""A small sample catalogue for demo mode (used when the ESCI data has not been built).

Every price, delivery time and stock flag below is SYNTHETIC, like the ones the real pipeline
generates for ESCI products. Rows: (title, brand, color, material, category, price, days, in_stock).
"""

from __future__ import annotations

_ROWS = [
    # running shoes
    ("Nike Revolution 7 Men's Road Running Shoes, Black", "Nike", "Black", "mesh", "Shoes", 54.99, 2, True),
    ("Nike Revolution 7 Women's Running Shoes, Red", "Nike", "Red", "mesh", "Shoes", 52.00, 2, True),
    ("Nike Downshifter 13 Women's Running Shoes, Red", "Nike", "Red", "mesh", "Shoes", 62.00, 3, True),
    ("Nike Pegasus 41 Men's Running Shoes, Blue", "Nike", "Blue", "mesh", "Shoes", 139.95, 2, True),
    ("Adidas Duramo SL Men's Running Shoes, Black", "Adidas", "Black", "mesh", "Shoes", 49.50, 4, True),
    ("Adidas Ultraboost Light Running Shoes, White", "Adidas", "White", "knit", "Shoes", 189.00, 3, True),
    ("Adidas Runfalcon 3.0 Women's Running Shoes, Pink", "Adidas", "Pink", "mesh", "Shoes", 44.99, 7, True),
    ("ASICS Gel-Contend 8 Men's Running Shoes, Black", "ASICS", "Black", "mesh", "Shoes", 58.00, 2, False),
    ("New Balance 520v8 Women's Running Shoes, Gray", "New Balance", "Gray", "mesh", "Shoes", 59.99, 5, True),
    ("Brooks Ghost 16 Men's Running Shoes, Black", "Brooks", "Black", "mesh", "Shoes", 139.00, 1, True),
    ("Crocs Classic Clog, Red", "Crocs", "Red", "rubber", "Shoes", 34.99, 2, True),
    # yoga & fitness
    ("Gaiam Essentials Yoga Mat, Purple, Non-Slip", "Gaiam", "Purple", "rubber", "Fitness", 24.99, 2, True),
    ("Manduka PRO Yoga Mat, Black, Extra Thick", "Manduka", "Black", "rubber", "Fitness", 132.00, 4, True),
    ("Amazon Basics Yoga Mat, Blue, 6mm Thick", "Amazon Basics", "Blue", "rubber", "Fitness", 19.50, 1, True),
    ("Lululemon Reversible Yoga Mat, Gray", "Lululemon", "Gray", "rubber", "Fitness", 88.00, 5, False),
    ("Yoga Bolster Pillow, Cotton Cover, Green", "Gaiam", "Green", "cotton", "Fitness", 39.00, 3, True),
    ("Neoprene Dumbbell Set 5 lb Pair, Pink", "Amazon Basics", "Pink", "neoprene", "Fitness", 18.99, 2, True),
    # bottles & kitchen
    ("Hydro Flask 32 oz Wide Mouth Water Bottle, Stainless Steel, Blue", "Hydro Flask", "Blue", "stainless steel", "Kitchen", 44.95, 2, True),
    ("Stanley Quencher 40 oz Tumbler, Stainless Steel, Black", "Stanley", "Black", "stainless steel", "Kitchen", 45.00, 3, True),
    ("Nalgene Wide Mouth Water Bottle 32 oz, Plastic, Green", "Nalgene", "Green", "plastic", "Kitchen", 15.99, 1, True),
    ("Contigo Glass Water Bottle 20 oz, Clear", "Contigo", "Clear", "glass", "Kitchen", 17.49, 4, True),
    ("Yeti Rambler 26 oz Bottle, Stainless Steel, Navy", "Yeti", "Navy", "stainless steel", "Kitchen", 40.00, 2, False),
    ("Lodge Cast Iron Skillet 12 inch, Black", "Lodge", "Black", "cast iron", "Kitchen", 29.90, 2, True),
    ("Ninja Air Fryer 4 Qt, Gray", "Ninja", "Gray", "plastic", "Kitchen", 99.99, 3, True),
    ("Bamboo Cutting Board Set of 3", "Utopia Kitchen", "Brown", "bamboo", "Kitchen", 21.99, 2, True),
    # audio & tech
    ("Sony WH-CH520 Wireless Headphones, Black", "Sony", "Black", "plastic", "Electronics", 58.00, 2, True),
    ("Sony WH-1000XM5 Noise Cancelling Headphones, Silver", "Sony", "Silver", "plastic", "Electronics", 328.00, 3, True),
    ("Apple AirPods Pro 2 Wireless Earbuds, White", "Apple", "White", "plastic", "Electronics", 189.00, 2, True),
    ("JBL Tune 510BT Wireless On-Ear Headphones, Blue", "JBL", "Blue", "plastic", "Electronics", 39.95, 1, True),
    ("JBL Flip 6 Portable Bluetooth Speaker, Red", "JBL", "Red", "plastic", "Electronics", 99.95, 4, True),
    ("Anker Soundcore Life Q30 Headphones, Black", "Anker", "Black", "plastic", "Electronics", 69.99, 2, False),
    ("Anker 20000mAh Portable Charger, Black", "Anker", "Black", "plastic", "Electronics", 35.99, 1, True),
    ("Logitech MX Master 3S Wireless Mouse, Graphite", "Logitech", "Gray", "plastic", "Electronics", 99.99, 2, True),
    # bags & apparel
    ("Herschel Little America Backpack, Black", "Herschel", "Black", "polyester", "Bags", 109.99, 3, True),
    ("JanSport SuperBreak Backpack, Navy", "JanSport", "Navy", "polyester", "Bags", 38.00, 2, True),
    ("Fossil Leather Messenger Bag, Brown", "Fossil", "Brown", "leather", "Bags", 148.00, 5, True),
    ("Madewell Leather Tote Bag, Tan", "Madewell", "Tan", "leather", "Bags", 168.00, 7, True),
    ("Levi's 501 Original Fit Men's Jeans, Blue", "Levi's", "Blue", "denim", "Apparel", 59.50, 3, True),
    ("Levi's Trucker Denim Jacket, Black", "Levi's", "Black", "denim", "Apparel", 89.50, 2, True),
    ("Hanes Men's Cotton T-Shirt 6-Pack, White", "Hanes", "White", "cotton", "Apparel", 21.99, 1, True),
    ("Carhartt Wool Beanie Hat, Gray", "Carhartt", "Gray", "wool", "Apparel", 19.99, 2, True),
    ("Patagonia Better Sweater Fleece Jacket, Red", "Patagonia", "Red", "fleece", "Apparel", 139.00, 4, True),
    ("Columbia Men's Fleece Jacket, Black", "Columbia", "Black", "fleece", "Apparel", 49.99, 2, True),
    ("Cashmere Scarf, Soft Wool Blend, Gray", "Naadam", "Gray", "cashmere", "Apparel", 95.00, 6, True),
    ("Nike Dri-FIT Men's Training T-Shirt, Black", "Nike", "Black", "polyester", "Apparel", 28.00, 2, True),
    # home
    ("Dyson V8 Cordless Vacuum Cleaner, Silver", "Dyson", "Silver", "plastic", "Home", 349.99, 3, True),
    ("Shark Navigator Lift-Away Vacuum, Blue", "Shark", "Blue", "plastic", "Home", 169.99, 2, True),
    ("Brooklinen Luxe Sateen Sheet Set, White, Cotton", "Brooklinen", "White", "cotton", "Home", 119.00, 5, True),
    ("Linen Duvet Cover Queen, Beige", "Brooklinen", "Beige", "linen", "Home", 99.00, 7, True),
    ("Philips Hue White Smart Bulb 4-Pack", "Philips", "White", "glass", "Home", 39.99, 1, True),
    ("Yankee Candle Large Jar, Ceramic, Red", "Yankee Candle", "Red", "ceramic", "Home", 27.99, 2, True),
    ("Ceramic Planter Pot 8 inch, White", "Mkono", "White", "ceramic", "Home", 16.99, 3, True),
    ("Wooden Desk Organizer, Walnut", "Simple Houseware", "Brown", "wood", "Home", 24.50, 4, True),
    # kids
    ("LEGO Classic Large Creative Brick Box, Multicolor", "LEGO", "Multicolor", "plastic", "Toys", 49.99, 2, True),
    ("Crayola 64 Count Crayons", "Crayola", "Multicolor", "wax", "Toys", 4.99, 1, True),
    ("Melissa & Doug Wooden Building Blocks", "Melissa & Doug", "Brown", "wood", "Toys", 21.99, 3, True),
    ("Nike Kids' Star Runner Running Shoes, Pink", "Nike", "Pink", "mesh", "Shoes", 42.00, 2, True),
]

CATALOG = [
    {
        "product_id": f"D{i:04d}",
        "title": title,
        "brand": brand,
        "color": color,
        "material": material,
        "category": category,
        "price": price,
        "delivery_days": days,
        "in_stock": stock,
    }
    for i, (title, brand, color, material, category, price, days, stock) in enumerate(_ROWS, 1)
]

EXAMPLES = [
    "red nike running shoes under $80 delivered in 2 days",
    "stainless steel water bottle under $50 in stock",
    "black wireless headphones under $70 within 3 days",
    "leather bag under $150",
    "cotton sheets, white, delivered in 5 days",
    "a yoga mat and a water bottle",
    "men's running shoes without red under $60",
    "fleece jacket under $100 in stock",
]
