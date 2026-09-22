from __future__ import annotations

import json
import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODEL_DIR
from src.database import query_df
from src.optimizer import drone_feasibility, nearest_neighbor_route
from src.services import agent_chat, summarize_reviews, track_order

st.set_page_config(
    page_title="SmartLogix AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif}.stApp{background:#f6f8fc;color:#13213c}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#061936,#082446);border-right:1px solid #17365e}
[data-testid="stSidebar"] *{color:#eaf1ff}[data-testid="stSidebar"] .stRadio label{padding:.55rem .7rem;border-radius:9px;margin:2px 0}
[data-testid="stSidebar"] .stRadio label:hover{background:#12345f}.block-container{padding:1.2rem 1.6rem 2rem;max-width:1500px}
h1{font-size:1.55rem!important;color:#12213c;margin-bottom:.2rem!important}h2,h3{color:#15233d}
.subtitle{color:#6b7b96;margin-bottom:1rem;font-size:.9rem}.card{background:white;border:1px solid #e5eaf2;border-radius:13px;padding:16px;box-shadow:0 2px 8px rgba(20,42,80,.04);margin-bottom:12px}
.kpi-label{color:#718096;font-size:.76rem;font-weight:600;text-transform:uppercase}.kpi-value{color:#14213d;font-size:1.55rem;font-weight:700;margin:.22rem 0}.kpi-note{color:#16a34a;font-size:.72rem}
.tip{background:#eef5ff;border-left:4px solid #2563eb;border-radius:8px;padding:10px 12px;color:#26466f;font-size:.84rem;margin:.5rem 0 1rem}
.status-green{display:inline-block;background:#e8f8ee;color:#18864b;padding:3px 9px;border-radius:999px;font-size:.72rem;font-weight:600}.status-blue{display:inline-block;background:#eaf2ff;color:#2563eb;padding:3px 9px;border-radius:999px;font-size:.72rem;font-weight:600}
div[data-testid="stMetric"]{background:white;border:1px solid #e5eaf2;padding:13px 15px;border-radius:12px;box-shadow:0 2px 8px rgba(20,42,80,.04)}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] p,div[data-testid="stMetric"] span,[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:#13213c!important;-webkit-text-fill-color:#13213c!important;opacity:1!important}
.stButton>button{border-radius:8px!important;border:1px solid #b9cdfb!important;font-weight:600!important;background:#ffffff!important;color:#2563eb!important}
.stButton>button:hover{background:#eef5ff!important;border-color:#2563eb!important;color:#174bb8!important}
.stButton>button p,.stButton>button span{color:#2563eb!important;-webkit-text-fill-color:#2563eb!important}
button[data-testid="stBaseButton-secondary"],button[kind="secondary"]{background:#ffffff!important;color:#2563eb!important;border:1px solid #b9cdfb!important}
button[data-testid="stBaseButton-secondary"] p,button[data-testid="stBaseButton-secondary"] span,button[kind="secondary"] p,button[kind="secondary"] span{color:#2563eb!important;-webkit-text-fill-color:#2563eb!important;opacity:1!important}
button[data-testid="stBaseButton-primary"],.stButton>button[kind="primary"]{background:#2563eb!important;color:#ffffff!important;border-color:#2563eb!important}
button[data-testid="stBaseButton-primary"] p,button[data-testid="stBaseButton-primary"] span,.stButton>button[kind="primary"] p,.stButton>button[kind="primary"] span{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
[data-testid="stDataFrame"]{border:1px solid #e5eaf2;border-radius:12px;overflow:hidden}
[data-testid="stAlert"] p,[data-testid="stAlert"] span{color:#13213c!important;-webkit-text-fill-color:#13213c!important}
[data-testid="stChatMessage"]{background:#ffffff!important;border:1px solid #dfe6f1!important;border-radius:12px!important;padding:12px!important;margin:8px 0!important}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li,[data-testid="stChatMessage"] span{color:#13213c!important}
[data-testid="stChatInput"]{background:#ffffff!important;border:1px solid #cbd5e1!important;border-radius:12px!important}
[data-testid="stChatInput"] textarea{color:#13213c!important;background:#ffffff!important;-webkit-text-fill-color:#13213c!important}
[data-testid="stChatInput"] textarea::placeholder{color:#718096!important;-webkit-text-fill-color:#718096!important}

/* Route Optimization form visibility */
div[data-testid="stNumberInput"] label p,
div[data-testid="stTextArea"] label p{
    color:#17233c!important;
    -webkit-text-fill-color:#17233c!important;
    opacity:1!important;
    font-weight:600!important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea{
    color:#ffffff!important;
    -webkit-text-fill-color:#ffffff!important;
    caret-color:#ffffff!important;
}

div[data-testid="stTextArea"] textarea::placeholder{
    color:#b8c0d0!important;
    -webkit-text-fill-color:#b8c0d0!important;
    opacity:1!important;
}
</style>""",
    unsafe_allow_html=True,
)


def title(text, description):
    st.title(text)
    st.markdown(f'<div class="subtitle">{description}</div>', unsafe_allow_html=True)


def tip(text):
    st.markdown(
        f'<div class="tip"><b>Tip:</b> {text}</div>', unsafe_allow_html=True
    )


def kpi(label, value, note, icon):
    st.markdown(
        f'<div class="card"><div class="kpi-label">{icon} &nbsp;{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def page_dashboard():
    title(
        "Dashboard",
        "A quick view of orders, delivery performance, revenue and fleet availability.",
    )
    totals = query_df(
        """SELECT COUNT(*) total,SUM(CASE WHEN LOWER(order_status)='delivered' THEN 1 ELSE 0 END) delivered,
    SUM(CASE WHEN LOWER(order_status)='in transit' THEN 1 ELSE 0 END) transit,
    SUM(CASE WHEN actual_delivery_hours>promised_eta_hours THEN 1 ELSE 0 END) delayed,SUM(order_value_inr) revenue FROM orders"""
    ).iloc[0]
    values = [
        ("Total Orders", f"{int(totals.total):,}", "All orders", "▣"),
        ("Delivered", f"{int(totals.delivered):,}", "Completed successfully", "✓"),
        ("In Transit", f"{int(totals.transit):,}", "Currently moving", "⇢"),
        ("Delayed", f"{int(totals.delayed):,}", "Past promised ETA", "◷"),
        (
            "Total Revenue",
            f"₹{float(totals.revenue or 0)/1e7:.2f} Cr",
            "Recorded order value",
            "₹",
        ),
    ]
    for col, item in zip(st.columns(5), values):
        with col:
            kpi(*item)
    left, middle, right = st.columns([1.45, 1, 0.9])
    monthly = query_df(
        "SELECT order_date,order_status FROM orders WHERE order_date IS NOT NULL"
    )
    monthly["order_date"] = pd.to_datetime(monthly["order_date"])
    monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
    overview = (
        monthly.groupby(["month", "order_status"])
        .size()
        .reset_index(name="orders")
        .sort_values("month")
        .tail(36)
    )
    mode = query_df(
        "SELECT transport_mode,COUNT(*) orders FROM orders WHERE transport_mode IS NOT NULL GROUP BY transport_mode"
    )
    recent = query_df(
        "SELECT order_id,transport_mode,order_status,destination_city FROM orders ORDER BY order_date DESC LIMIT 7"
    )
    with left:
        st.subheader("Delivery overview")
        st.plotly_chart(
            px.line(
                overview, x="month", y="orders", color="order_status", markers=True
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with middle:
        st.subheader("Deliveries by mode")
        st.plotly_chart(
            px.pie(
                mode,
                names="transport_mode",
                values="orders",
                hole=0.58,
                color_discrete_sequence=[
                    "#2563eb",
                    "#2dd4bf",
                    "#f59e0b",
                    "#fb7185",
                    "#8b5cf6",
                    "#38bdf8",
                ],
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.subheader("Recent orders")
        st.dataframe(recent, hide_index=True, width="stretch", height=285)
    st.subheader("Fleet status")
    fleet = query_df(
        "SELECT vehicle_type,COUNT(*) total,SUM(CASE WHEN LOWER(fleet_status) IN ('active','in service') THEN 1 ELSE 0 END) active FROM fleet GROUP BY vehicle_type ORDER BY total DESC"
    )
    icons = {
        "Drone": "✈",
        "Bike": "◉",
        "Van": "▰",
        "Truck": "▣",
        "Air Cargo": "⌁",
        "Ship": "≈",
    }
    for col, row in zip(st.columns(len(fleet)), fleet.itertuples()):
        with col:
            kpi(
                row.vehicle_type,
                str(row.total),
                f"{int(row.active or 0)} active",
                icons.get(row.vehicle_type, "●"),
            )


def page_orders():
    title(
        "My Orders", "Search, filter and inspect customer orders without writing SQL."
    )
    tip("Start with All Statuses. Search using an order ID, city or transport mode.")
    a, b = st.columns([1, 2])
    statuses = ["All Statuses"] + query_df(
        "SELECT DISTINCT order_status FROM orders WHERE order_status IS NOT NULL ORDER BY order_status"
    )["order_status"].tolist()
    status = a.selectbox("Order status", statuses)
    search = b.text_input("Search orders", placeholder="ORD-00125, Bengaluru, Drone")
    df = query_df(
        "SELECT order_id,order_date,destination_city,quantity,transport_mode,order_status,order_value_inr,promised_eta_hours,actual_delivery_hours FROM orders ORDER BY order_date DESC"
    )
    if status != "All Statuses":
        df = df[df.order_status == status]
    if search:
        df = df[
            df.astype(str)
            .apply(lambda c: c.str.contains(search, case=False, na=False))
            .any(axis=1)
        ]
    st.caption(f"Showing {min(len(df),500):,} of {len(df):,} matching orders")
    st.dataframe(
        df.head(500),
        hide_index=True,
        width="stretch",
        column_config={
            "order_value_inr": st.column_config.NumberColumn("Amount", format="₹ %.2f")
        },
    )


def page_tracking():
    title(
        "Delivery Tracking",
        "Enter an order ID to see its current status and shipment journey.",
    )
    tip("Try an order ID from My Orders, such as ORD-000001.")
    c1, c2 = st.columns([4, 1])
    oid = c1.text_input("Order ID", value="ORD-000001").strip().upper()
    clicked = c2.button("Track delivery", type="primary", width="stretch")
    if clicked or oid:
        result = track_order(oid)
        if not result["found"]:
            st.warning("Order not found. Check the ID and try again.")
            return
        order = result["order"]
        left, right = st.columns([1, 2.2])
        with left:
            st.subheader("Order details")
            st.write(f"**Order ID:** {order['order_id']}")
            st.markdown(
                f"**Status:** <span class='status-blue'>{order['order_status']}</span>",
                unsafe_allow_html=True,
            )
            st.write(
                f"**Delivery mode:** {order.get('transport_mode') or 'Not assigned'}"
            )
            st.write(f"**Destination:** {order.get('destination_city')}")
            st.write(f"**Promised ETA:** {order.get('promised_eta_hours')} hours")
        with right:
            st.subheader("Tracking timeline")
            events = pd.DataFrame(result["events"])
            if events.empty:
                st.info("No scan events are available yet.")
            else:
                st.dataframe(events, hide_index=True, width="stretch")
        location = query_df(
            "SELECT destination_lat,destination_lon FROM orders WHERE order_id=:id",
            {"id": oid},
        )
        if not location.empty:
            location["destination_lat"] = pd.to_numeric(
                location["destination_lat"], errors="coerce"
            )
            location["destination_lon"] = pd.to_numeric(
                location["destination_lon"], errors="coerce"
            )
            location = location.dropna(subset=["destination_lat", "destination_lon"])
        if not location.empty:
            st.subheader("Destination map")
            map_data = location.rename(
                columns={"destination_lat": "lat", "destination_lon": "lon"}
            )
            st.map(map_data[["lat", "lon"]], zoom=10)


def page_mode():
    title(
        "Choose Delivery Mode",
        "Use the trained model to recommend a suitable transportation mode.",
    )
    tip(
        "Fill the form like a new order. The model learned from historical SmartLogix orders."
    )
    with st.form("mode_form"):
        c1, c2, c3 = st.columns(3)
        distance = c1.number_input("Distance (km)", 0.1, value=16.8)
        weight = c1.number_input("Package weight (kg)", 0.01, value=2.4)
        quantity = c1.number_input("Quantity", 1, value=1)
        priority = c2.selectbox(
            "Delivery priority", ["Standard", "Express", "Economy", "Same Day"]
        )
        weather = c2.selectbox(
            "Weather", ["Clear", "Cloudy", "Rain", "Heavy Rain", "Fog", "Thunderstorm"]
        )
        origin = c2.text_input("Origin city", value="Bengaluru")
        destination = c3.text_input("Destination city", value="Mysuru")
        value = c3.number_input("Order value (₹)", 0.0, value=5000.0)
        fragile = c3.checkbox("Fragile package")
        hazmat = c3.checkbox("Hazardous material")
        cold = c3.checkbox("Cold-chain required")
        submitted = st.form_submit_button("Recommend mode", type="primary")
    if submitted:
        path = MODEL_DIR / "mode_classifier.joblib"
        if not path.exists():
            st.error("Run: python scripts/train_models.py")
        else:
            row = pd.DataFrame(
                [
                    {
                        "distance_km": distance,
                        "package_weight_kg": weight,
                        "quantity": quantity,
                        "is_fragile": int(fragile),
                        "is_hazmat": int(hazmat),
                        "cold_chain_required": int(cold),
                        "order_value_inr": value,
                        "delivery_priority": priority,
                        "weather_condition_at_dest": weather,
                        "origin_city": origin,
                        "destination_city": destination,
                    }
                ]
            )
            model = joblib.load(path)
            prediction = model.predict(row)[0]
            prob = max(model.predict_proba(row)[0])
            st.success(
                f"Recommended mode: **{prediction}** — model confidence {prob:.1%}"
            )
            st.caption(
                "Use this as decision support. Safety, availability and business rules must still be checked."
            )


def page_route():
    title(
        "Route Optimization",
        "Arrange delivery stops using an explainable nearest-neighbour baseline.",
    )
    tip(
        "Coordinates are latitude and longitude. Add 2–5 stops for a simple demonstration."
    )
    a, b = st.columns([1, 2])
    with a:
        olat = st.number_input("Warehouse latitude", value=12.9716, format="%.6f")
        olon = st.number_input("Warehouse longitude", value=77.5946, format="%.6f")
        text = st.text_area(
            "Stops: name, latitude, longitude",
            value="Customer A,12.9352,77.6245\nCustomer B,12.9698,77.7500\nCustomer C,13.0358,77.5970",
            height=130,
        )
        run = st.button("Optimize route", type="primary", width="stretch")
    if run:
        try:
            stops = []
            for line in text.splitlines():
                name, lat, lon = [x.strip() for x in line.split(",")]
                stops.append({"name": name, "lat": float(lat), "lon": float(lon)})
            result = nearest_neighbor_route((olat, olon), stops)
            points = pd.DataFrame(
                [{"name": "Warehouse", "lat": olat, "lon": olon}]
                + result["ordered_stops"]
                + [{"name": "Warehouse", "lat": olat, "lon": olon}]
            )
            with b:
                st.metric("Optimized distance", f"{result['total_distance_km']} km")
                st.map(points[["lat", "lon"]], zoom=10)
            st.dataframe(
                pd.DataFrame(result["ordered_stops"])[
                    ["name", "lat", "lon", "leg_distance_km"]
                ],
                hide_index=True,
                width="stretch",
            )
        except ValueError:
            st.error("Use: Customer A,12.9352,77.6245")


def page_fleet():
    title("Fleet Management", "Monitor vehicle availability, ownership and capacity.")
    tip(
        "Active vehicles are ready; Maintenance vehicles should not receive a new route."
    )
    fleet = query_df(
        "SELECT vehicle_id,vehicle_type,model_name,capacity_kg,max_range_km,hub_code,fleet_status,ownership FROM fleet"
    )
    a, b, c = st.columns(3)
    a.metric("Total vehicles", f"{len(fleet):,}")
    b.metric(
        "Active / in service",
        f"{fleet.fleet_status.str.lower().isin(['active','in service']).sum():,}",
    )
    c.metric(
        "Under maintenance",
        f"{fleet.fleet_status.str.lower().eq('maintenance').sum():,}",
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                fleet.groupby(["vehicle_type", "fleet_status"])
                .size()
                .reset_index(name="vehicles"),
                x="vehicle_type",
                y="vehicles",
                color="fleet_status",
                barmode="stack",
                title="Vehicles by status",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            px.pie(fleet, names="ownership", title="Fleet ownership", hole=0.55),
            width="stretch",
            config={"displayModeBar": False},
        )
    st.dataframe(fleet, hide_index=True, width="stretch")


def page_drone():
    title(
        "Drone Health & Inspection",
        "Explore sensors and check whether a drone can safely complete a delivery.",
    )
    tip(
        "High temperature/vibration and low battery health can indicate maintenance risk."
    )
    health = query_df(
        "SELECT drone_id,battery_health_pct,motor_temp_c,vibration_rms,payload_kg,wind_speed_kmph,gps_signal_quality,maintenance_required FROM drone_telemetry"
    )
    for column in [
        "battery_health_pct",
        "motor_temp_c",
        "vibration_rms",
        "payload_kg",
        "wind_speed_kmph",
        "maintenance_required",
    ]:
        health[column] = pd.to_numeric(health[column], errors="coerce")
    health = health.dropna(
        subset=["battery_health_pct", "motor_temp_c", "vibration_rms"]
    )
    health["vibration_rms"] = health["vibration_rms"].clip(lower=0)
    a, b, c = st.columns(3)
    a.metric("Flights analysed", f"{len(health):,}")
    b.metric("Maintenance alerts", f"{int(health.maintenance_required.sum()):,}")
    c.metric("Average battery health", f"{health.battery_health_pct.mean():.1f}%")
    left, right = st.columns([1.4, 1])
    sample = health.sample(min(3000, len(health)), random_state=42)
    with left:
        st.plotly_chart(
            px.scatter(
                sample,
                x="battery_health_pct",
                y="motor_temp_c",
                color="maintenance_required",
                size="vibration_rms",
                hover_data=["drone_id"],
                title="Battery health vs motor temperature",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.subheader("Drone feasibility check")
        distance = st.number_input("Round-trip distance (km)", 0.1, value=8.0)
        payload = st.number_input("Payload (kg)", 0.01, value=1.5)
        max_range = st.number_input("Maximum range (km)", 1.0, value=20.0)
        capacity = st.number_input("Payload capacity (kg)", 0.1, value=3.0)
        battery = st.slider("Starting battery (%)", 0, 100, 90)
        if st.button("Check feasibility", type="primary"):
            result = drone_feasibility(distance, payload, max_range, capacity, battery)
            if result["feasible"]:
                st.success(f"Feasible. Usable range: {result['usable_range_km']} km")
            else:
                st.error("Not feasible: " + "; ".join(result["reasons"]))
    st.subheader("Recent sensor records")
    st.dataframe(health.head(250), hide_index=True, width="stretch")


def page_analytics():
    title(
        "Analytics", "Understand delivery cost, speed, delays and customer experience."
    )
    orders = query_df(
        "SELECT transport_mode,distance_km,delivery_cost_inr,promised_eta_hours,actual_delivery_hours,on_time FROM orders WHERE transport_mode IS NOT NULL"
    )
    for c in [
        "distance_km",
        "delivery_cost_inr",
        "promised_eta_hours",
        "actual_delivery_hours",
        "on_time",
    ]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce")

    orders["timing_variance_hours"] = (
        orders["actual_delivery_hours"] - orders["promised_eta_hours"]
    )

    a, b = st.columns(2)
    with a:
        st.plotly_chart(
            px.box(
                orders,
                x="transport_mode",
                y="timing_variance_hours",
                color="transport_mode",
                title="Delivery Timing Variance by Mode",
                labels={
                    "transport_mode": "Transport Mode",
                    "timing_variance_hours": "Timing Variance (Hours)",
                },
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    with b:
        st.plotly_chart(
            px.scatter(
                orders.sample(min(3500, len(orders)), random_state=42),
                x="distance_km",
                y="delivery_cost_inr",
                color="transport_mode",
                opacity=0.55,
                title="Distance and Delivery Cost",
                labels={
                    "distance_km": "Distance (km)",
                    "delivery_cost_inr": "Delivery Cost (INR)",
                    "transport_mode": "Transport Mode",
                },
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    summary = (
        orders.groupby("transport_mode")
        .agg(
            orders=("transport_mode", "size"),
            average_cost=("delivery_cost_inr", "mean"),
            average_duration=("actual_delivery_hours", "mean"),
            on_time_pct=("on_time", "mean"),
        )
        .reset_index()
    )
    summary.on_time_pct *= 100
    st.dataframe(summary, hide_index=True, width="stretch")


def format_assistant_response(agent_result):
    """Convert agent tool output into a clear answer for the chat screen."""
    agent = agent_result.get("agent")
    result = agent_result.get("result", {})

    if agent == "support_rag":
        return result.get(
            "answer",
            "I could not find a reliable answer in the current knowledge base.",
        )

    if agent == "order_tracking":
        if not result.get("found"):
            return result.get("message", "I could not find that order.")

        order = result.get("order", {})
        events = result.get("events", [])
        answer = (
            f"Order {order.get('order_id')} is currently "
            f"{order.get('order_status', 'unavailable')}.\n\n"
            f"Delivery mode: {order.get('transport_mode') or 'Not assigned'}\n\n"
            f"Destination: {order.get('destination_city') or 'Unavailable'}\n\n"
            f"Promised ETA: {order.get('promised_eta_hours') or 'Unavailable'} hours"
        )
        if events:
            latest = events[-1]
            answer += (
                f"\n\nLatest update: {latest.get('event_type', 'Update')}"
                f" at {latest.get('location_city') or 'an unavailable location'}."
            )
        return answer

    if agent == "product_recommendation":
        if not result:
            return "I could not find any available products to recommend."
        lines = ["Here are some highly rated products:"]
        for product in result[:5]:
            price = product.get("price_inr")
            price_text = (
                f"₹{float(price):,.2f}"
                if price not in (None, "")
                else "Price unavailable"
            )
            lines.append(
                f"- {product.get('product_name')} ({product.get('product_id')}) — "
                f"{price_text}, rating {product.get('avg_rating') or 'N/A'}"
            )
        return "\n".join(lines)

    if agent == "review_analysis":
        if not result.get("found"):
            return result.get("message", "No reviews were found for that product.")
        counts = result.get("sentiment_counts", {})
        return (
            f"I found {result.get('review_count', 0)} reviews. "
            f"The average rating is {result.get('average_rating', 'N/A')} out of 5.\n\n"
            f"Positive: {counts.get('pos', 0)} | "
            f"Neutral: {counts.get('neu', 0)} | "
            f"Negative: {counts.get('neg', 0)}"
        )

    return (
        "I could not understand that request. Try asking about an order, "
        "product, review, or delivery question."
    )


def get_chatbot_answer(query):
    """Answer common platform questions, then use the SmartLogix agent router."""
    clean_query = query.strip()
    lower_query = clean_query.lower()

    if lower_query in {"hi", "hello", "hey", "good morning", "good evening"}:
        return (
            "Hello! I can track an order, recommend products, summarize reviews, "
            "explain delivery delays, and answer questions about SmartLogix operations."
        )

    if "what can you do" in lower_query or lower_query == "help":
        return (
            "I can help with:\n"
            "- Tracking an order using an ID such as ORD-000001\n"
            "- Recommending available products\n"
            "- Summarizing reviews for a product ID\n"
            "- Explaining delivery delays and drone delivery\n"
            "- Reporting basic order, fleet, and maintenance totals"
        )

    if "how many orders" in lower_query or "total orders" in lower_query:
        total = query_df("SELECT COUNT(*) AS total FROM orders").iloc[0]["total"]
        return f"There are {int(total):,} orders in the SmartLogix database."

    if "delivery modes" in lower_query or "transport modes" in lower_query:
        modes = query_df(
            "SELECT transport_mode, COUNT(*) AS orders FROM orders "
            "WHERE transport_mode IS NOT NULL GROUP BY transport_mode "
            "ORDER BY orders DESC"
        )
        lines = ["The available transportation modes and recorded order counts are:"]
        for row in modes.itertuples():
            lines.append(f"- {row.transport_mode}: {int(row.orders):,} orders")
        return "\n".join(lines)

    if "maintenance alerts" in lower_query or "maintenance required" in lower_query:
        total = query_df(
            "SELECT COUNT(*) AS total FROM drone_telemetry "
            "WHERE maintenance_required = 1"
        ).iloc[0]["total"]
        return f"The drone telemetry data contains {int(total):,} maintenance alerts."

    product_ids = re.findall(r"PRD-\d+", clean_query.upper())
    if "compare" in lower_query and len(product_ids) >= 2:
        first_id, second_id = product_ids[:2]
        products = query_df(
            "SELECT product_id, product_name, category, price_inr, avg_rating, stock_qty "
            "FROM products WHERE product_id = :first_id OR product_id = :second_id",
            {"first_id": first_id, "second_id": second_id},
        )
        if len(products) < 2:
            return "I could not find both product IDs. Please check them and try again."
        lines = ["Product comparison:"]
        for row in products.itertuples():
            lines.append(
                f"- {row.product_name} ({row.product_id}): ₹{float(row.price_inr):,.2f}, "
                f"rating {row.avg_rating or 'N/A'}, stock {row.stock_qty}"
            )
        return "\n".join(lines)

    return format_assistant_response(agent_chat(clean_query))


def page_assistant():
    title(
        "AI Assistant",
        "Ask about order tracking, product recommendations, reviews or delivery FAQs.",
    )
    tip("Try: Track ORD-000001, Recommend products, or Why is delivery delayed?")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    examples = [
        "Track ORD-000001",
        "Recommend products",
        "Why is delivery delayed?",
        "How does drone delivery work?",
    ]
    selected = None
    for col, prompt in zip(st.columns(4), examples):
        if col.button(prompt, width="stretch"):
            selected = prompt
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    query = st.chat_input("Type your message...") or selected
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        response = get_chatbot_answer(query)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        st.rerun()


def page_products():
    title(
        "Products",
        "Browse available products and receive rating-based recommendations.",
    )
    categories = ["All Categories"] + query_df(
        "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
    ).category.tolist()
    a, b = st.columns([1, 2])
    category = a.selectbox("Category", categories)
    max_price = b.slider("Maximum price (₹)", 500, 150000, 100000, 500)
    products = query_df(
        "SELECT product_id,product_name,category,sub_category,price_inr,avg_rating,stock_qty FROM products WHERE stock_qty>0 ORDER BY avg_rating DESC"
    )
    products.price_inr = pd.to_numeric(products.price_inr, errors="coerce")
    if category != "All Categories":
        products = products[products.category == category]
    products = products[products.price_inr <= max_price]
    st.caption(f"{len(products):,} products match your filters")
    st.dataframe(
        products.head(200),
        hide_index=True,
        width="stretch",
        column_config={
            "price_inr": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "avg_rating": st.column_config.ProgressColumn(
                "Rating", min_value=0, max_value=5, format="%.1f ⭐"
            ),
        },
    )


def page_reviews():
    title(
        "Reviews & Insights", "Summarize ratings and customer sentiment for a product."
    )
    tip("Copy a Product ID from Products, such as PRD-00001.")
    pid = st.text_input("Product ID", value="PRD-00001").upper().strip()
    if st.button("Analyse reviews", type="primary"):
        result = summarize_reviews(pid)
        if not result["found"]:
            st.warning(result["message"])
        else:
            a, b = st.columns(2)
            a.metric("Reviews", result["review_count"])
            b.metric("Average rating", f"{result['average_rating']} / 5")
            sentiment = pd.DataFrame(
                list(result["sentiment_counts"].items()),
                columns=["sentiment", "reviews"],
            )
            st.plotly_chart(
                px.bar(
                    sentiment,
                    x="sentiment",
                    y="reviews",
                    color="sentiment",
                    color_discrete_map={
                        "pos": "#16a34a",
                        "neu": "#f59e0b",
                        "neg": "#ef4444",
                    },
                ),
                width="stretch",
            )
            for comment in result["sample_comments"]:
                st.info(comment)


def page_notifications():
    title(
        "Notifications",
        "Operational alerts from recent delivery events and exceptions.",
    )
    alerts = query_df(
        "SELECT order_id,event_type,event_timestamp,location_city,remarks,exception_code FROM delivery_logs WHERE exception_code IS NOT NULL OR event_type IN ('OUT_FOR_DELIVERY','DELIVERED') ORDER BY event_timestamp DESC LIMIT 100"
    )
    if alerts.empty:
        st.info("There are no recent alerts.")
    else:
        st.dataframe(alerts, hide_index=True, width="stretch")


def page_models():
    title(
        "Model Performance",
        "Complete evaluation results for classification and regression models.",
    )
    tip(
        "Classification and regression solve different problems, so their scores "
        "are displayed separately instead of being combined into one percentage."
    )
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        st.warning("Run: python scripts/train_models.py")
        return

    m = json.loads(path.read_text())

    mode = m["mode_classifier"]
    maintenance = m["maintenance_classifier"]
    sentiment = m["sentiment_classifier"]
    eta = m["eta_regressor"]

    classification_rows = [
        {
            "Model": "Transport Mode Classification",
            "Accuracy": mode["accuracy"],
            "Weighted F1": mode["f1_weighted"],
            "ROC-AUC": mode.get("roc_auc_ovr_weighted"),
        },
        {
            "Model": "Predictive Maintenance",
            "Accuracy": maintenance["accuracy"],
            "Weighted F1": maintenance["f1_weighted"],
            "ROC-AUC": maintenance.get("roc_auc_ovr_weighted"),
        },
        {
            "Model": "Review Sentiment Classification",
            "Accuracy": sentiment["accuracy"],
            "Weighted F1": sentiment["f1_weighted"],
            "ROC-AUC": sentiment.get("roc_auc_ovr_weighted"),
        },
    ]
    classification_df = pd.DataFrame(classification_rows)
    average_accuracy = classification_df["Accuracy"].mean()
    average_f1 = classification_df["Weighted F1"].mean()

    summary_cards = st.columns(4)
    summary_cards[0].metric(
        "Classification average accuracy",
        f"{average_accuracy:.1%}",
        help="Arithmetic mean of the three classification model accuracies.",
    )
    summary_cards[1].metric(
        "Classification average F1",
        f"{average_f1:.1%}",
        help="Arithmetic mean of the three weighted F1 scores.",
    )
    summary_cards[2].metric("ETA regression R²", f"{eta['r2']:.1%}")
    summary_cards[3].metric("ETA regression RMSE", f"{eta['rmse']:.2f} h")

    classification_tab, regression_tab, details_tab = st.tabs(
        ["Classification Models", "Regression Model", "Complete Metrics"]
    )

    with classification_tab:
        st.subheader("Classification model comparison")
        display_classification = classification_df.copy()
        for column in ["Accuracy", "Weighted F1", "ROC-AUC"]:
            display_classification[column] = display_classification[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else "Not available"
            )
        st.dataframe(
            display_classification,
            hide_index=True,
            width="stretch",
        )

        chart_data = classification_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Weighted F1", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        classification_chart = px.bar(
            chart_data,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".1%",
            title="Classification performance comparison",
            color_discrete_sequence=["#2563eb", "#16a34a", "#f59e0b"],
        )
        classification_chart.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(
            classification_chart,
            width="stretch",
            config={"displayModeBar": False},
        )

        st.warning(
            "Transport-mode accuracy is limited by imbalanced and inconsistent "
            "historical labels. Sentiment performance is unusually high because "
            "the simulated review phrases repeat frequently."
        )

    with regression_tab:
        st.subheader("ETA regression performance")
        regression_cards = st.columns(4)
        regression_cards[0].metric("R² score", f"{eta['r2']:.2%}")
        regression_cards[1].metric("MAE", f"{eta['mae']:.2f} hours")
        regression_cards[2].metric("MSE", f"{eta['mse']:.2f}")
        regression_cards[3].metric("RMSE", f"{eta['rmse']:.2f} hours")

        regression_table = pd.DataFrame(
            [
                {
                    "Model": "Delivery ETA Regression",
                    "R²": f"{eta['r2']:.2%}",
                    "MAE (hours)": round(eta["mae"], 3),
                    "MSE": round(eta["mse"], 3),
                    "RMSE (hours)": round(eta["rmse"], 3),
                }
            ]
        )
        st.dataframe(regression_table, hide_index=True, width="stretch")
        st.info(
            "R² shows how much variation in delivery time is explained by the model. "
            "MAE means the prediction differs from the actual delivery time by about "
            f"{eta['mae']:.2f} hours on average."
        )

    with details_tab:
        st.subheader("Complete technical metrics")
        st.json(m)


PAGES = {
    "▦  Dashboard": page_dashboard,
    "▤  Orders": page_orders,
    "⌖  Delivery Tracking": page_tracking,
    "⇄  Choose Delivery Mode": page_mode,
    "♧  Fleet Management": page_fleet,
    "⌁  Route Optimization": page_route,
    "✣  Drone Health": page_drone,
    "▥  Analytics": page_analytics,
    "◉  AI Assistant": page_assistant,
    "◇  Products": page_products,
    "☆  Reviews & Insights": page_reviews,
    "♢  Notifications": page_notifications,
    "⚙  Model Performance": page_models,
}

with st.sidebar:
    st.markdown("## ◉ SmartLogix AI")
    st.caption("Intelligent Logistics Platform")
    selected = st.radio("Navigation", list(PAGES), label_visibility="collapsed")

try:
    PAGES[selected]()
except Exception as exc:
    st.error("This page could not be displayed. Please check the error below.")
    st.code(str(exc))
