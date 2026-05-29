import streamlit as st

# App title
st.title("🍴 Online Food Ordering App")

# Menu data
menu = {
    "Pizza": 250,
    "Burger": 150,
    "Pasta": 200,
    "Sandwich": 120,
    "Coffee": 80
}

# Initialize session state for cart
if "cart" not in st.session_state:
    st.session_state.cart = {}

st.header("📋 Menu")

# Display menu
for item, price in menu.items():
    col1, col2, col3 = st.columns([1, 1, 1])

    col1.write(f"**{item}**")
    col2.write(f"₹{price}")

    if col3.button(f"Add {item}"):
        if item in st.session_state.cart:
            st.session_state.cart[item] += 1
        else:
            st.session_state.cart[item] = 1

# Show cart
st.header("🛒 Your Cart")

total = 0

if st.session_state.cart:
    for item, qty in st.session_state.cart.items():
        price = menu[item]
        subtotal = price * qty
        total += subtotal

        st.write(f"{item} x {qty} = ₹{subtotal}")

    st.subheader(f"Total: ₹{total}")

    if st.button("Place Order"):
        st.success("✅ Your order has been placed!")
        st.session_state.cart = {}

else:
    st.write("Your cart is empty.")

# Footer
st.markdown("---")
st.write("🍽️ Thank you for ordering!")