# food_ordering_system.py
# Advanced Food Ordering Management System (Console-Based)
import json
import os

# File paths
MENU_FILE = "menu.json"
ORDERS_FILE = "orders.json"

# Default menu (if menu.json doesn't exist)
default_menu = {
    1: {"name": "Pizza", "price": 250},
    2: {"name": "Burger", "price": 120},
    3: {"name": "Pasta", "price": 200},
    4: {"name": "Coke", "price": 50},
}

# Load or create menu
if os.path.exists(MENU_FILE):
    with open(MENU_FILE, "r") as f:
        menu = json.load(f)
else:
    menu = default_menu
    with open(MENU_FILE, "w") as f:
        json.dump(menu, f)

# Load or create orders
if os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "r") as f:
        orders = json.load(f)
else:
    orders = []
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f)

# ---------------- Functions ----------------

def show_menu():
    print("\n--- Menu ---")
    for item_id, item in menu.items():
        print(f"{item_id}. {item['name']} - ₹{item['price']}")

def place_order():
    order = []
    while True:
        show_menu()
        choice = input("Enter item number to order (or 'q' to finish): ")
        if choice.lower() == 'q':
            break
        elif choice.isdigit() and int(choice) in menu:
            quantity = int(input("Enter quantity: "))
            order.append({
                "item": menu[str(int(choice))]["name"], 
                "price": menu[str(int(choice))]["price"], 
                "quantity": quantity
            })
            print(f"{menu[str(int(choice))]['name']} x{quantity} added to order.")
        else:
            print("Invalid choice. Try again.")
    
    if order:
        total = sum(item["price"] * item["quantity"] for item in order)
        orders.append({"order_items": order, "total": total, "status": "Pending"})
        with open(ORDERS_FILE, "w") as f:
            json.dump(orders, f)
        print(f"\nOrder placed successfully! Total: ₹{total}")
    else:
        print("No items ordered.")

def view_orders():
    print("\n--- All Orders ---")
    if not orders:
        print("No orders yet.")
        return
    for i, order in enumerate(orders, start=1):
        print(f"\nOrder {i}:")
        for item in order["order_items"]:
            print(f"{item['item']} x{item['quantity']} - ₹{item['price']*item['quantity']}")
        print(f"Total: ₹{order['total']} | Status: {order['status']}")

def update_order_status():
    view_orders()
    if not orders:
        return
    order_num = int(input("Enter order number to update status: "))
    if 1 <= order_num <= len(orders):
        new_status = input("Enter new status (Pending/Preparing/Delivered): ")
        orders[order_num-1]["status"] = new_status
        with open(ORDERS_FILE, "w") as f:
            json.dump(orders, f)
        print(f"Order {order_num} status updated to {new_status}.")
    else:
        print("Invalid order number.")

def add_menu_item():
    item_name = input("Enter item name: ")
    item_price = float(input("Enter item price: "))
    new_id = str(max([int(k) for k in menu.keys()]) + 1)
    menu[new_id] = {"name": item_name, "price": item_price}
    with open(MENU_FILE, "w") as f:
        json.dump(menu, f)
    print(f"{item_name} added to menu successfully!")

def main():
    while True:
        print("\n--- Food Ordering Management System ---")
        print("1. Show Menu")
        print("2. Place Order")
        print("3. View Orders")
        print("4. Update Order Status")
        print("5. Add Menu Item (Admin)")
        print("6. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            show_menu()
        elif choice == "2":
            place_order()
        elif choice == "3":
            view_orders()
        elif choice == "4":
            update_order_status()
        elif choice == "5":
            add_menu_item()
        elif choice == "6":
            print("Exiting system. Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

# ---------------- Run the System ----------------
if __name__ == "__main__":
    main()
