-- PostgreSQL production schema. The local pipeline uses SQLite for zero-setup development.
CREATE TABLE IF NOT EXISTS customers (
  customer_id VARCHAR(20) PRIMARY KEY, customer_name TEXT, email TEXT, phone TEXT,
  city TEXT, state VARCHAR(10), pincode VARCHAR(12), signup_date TIMESTAMP,
  customer_segment TEXT, is_prime_member BOOLEAN, lifetime_orders INTEGER, avg_rating_given NUMERIC
);
CREATE TABLE IF NOT EXISTS products (
  product_id VARCHAR(20) PRIMARY KEY, product_name TEXT NOT NULL, category TEXT,
  sub_category TEXT, weight_kg NUMERIC, price_inr NUMERIC, stock_qty INTEGER,
  avg_rating NUMERIC, is_fragile BOOLEAN, is_hazmat BOOLEAN, requires_cold_chain BOOLEAN
);
CREATE TABLE IF NOT EXISTS fleet (
  vehicle_id VARCHAR(20) PRIMARY KEY, vehicle_type TEXT, model_name TEXT,
  capacity_kg NUMERIC, max_range_km NUMERIC, avg_speed_kmph NUMERIC,
  hub_code TEXT, fleet_status TEXT, battery_capacity_wh NUMERIC
);
CREATE TABLE IF NOT EXISTS orders (
  order_id VARCHAR(20) PRIMARY KEY, customer_id VARCHAR(20) REFERENCES customers(customer_id),
  product_id VARCHAR(20) REFERENCES products(product_id), assigned_vehicle_id VARCHAR(20) REFERENCES fleet(vehicle_id),
  order_date TIMESTAMP, distance_km NUMERIC, package_weight_kg NUMERIC,
  promised_eta_hours NUMERIC, actual_delivery_hours NUMERIC, delivery_cost_inr NUMERIC,
  transport_mode TEXT, order_status TEXT
);
CREATE TABLE IF NOT EXISTS delivery_logs (
  log_id VARCHAR(30) PRIMARY KEY, order_id VARCHAR(20) REFERENCES orders(order_id),
  event_seq INTEGER, event_type TEXT, event_timestamp TIMESTAMP, location_city TEXT, remarks TEXT
);
CREATE TABLE IF NOT EXISTS drone_telemetry (
  flight_id VARCHAR(30) PRIMARY KEY, drone_id VARCHAR(20) REFERENCES fleet(vehicle_id),
  flight_timestamp TIMESTAMP, battery_health_pct NUMERIC, motor_temp_c NUMERIC,
  vibration_rms NUMERIC, payload_kg NUMERIC, maintenance_required BOOLEAN
);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS ix_delivery_logs_order ON delivery_logs(order_id,event_timestamp);
CREATE INDEX IF NOT EXISTS ix_telemetry_drone ON drone_telemetry(drone_id,flight_timestamp);

