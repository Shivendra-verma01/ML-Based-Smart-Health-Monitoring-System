# =============================================
# ADVANCED ML BASED SMART HEALTH MONITORING SYSTEM
# Full Flask Application with Advanced Database Schema
# =============================================

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from functools import wraps
import mysql.connector
from mysql.connector import Error
import re
import hashlib
import secrets

#Machine Learning Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================
# APP CONFIGURATION
# =============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production-12345'
CORS(app, supports_credentials=True)
bcrypt = Bcrypt(app)

# Email Configuration (Optional)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
mail = Mail(app)

# =============================================
# DATABASE CONFIGURATION
# =============================================

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'shivendraverma7704085351',
    'database': 'health_monitoring',
    'autocommit': True,
    'use_pure': True
}

def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

# =============================================
# HELPER FUNCTIONS
# =============================================

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.generate_password_hash(password).decode('utf-8')

def verify_password(stored_hash, password):
    """Verify password"""
    return bcrypt.check_password_hash(stored_hash, password)

def login_required(f):
    """Decorator for login required routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# =============================================
# MACHINE LEARNING MODEL
# =============================================

class HealthRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=2000):
        """Generate synthetic training data for advanced model"""
        np.random.seed(42)
        data = []
        for _ in range(n_samples):
            hr = np.random.normal(75, 15)
            sbp = np.random.normal(120, 15)
            dbp = np.random.normal(80, 10)
            spo2 = np.random.normal(97, 2)
            temp = np.random.normal(98.6, 0.5)
            stress = np.random.normal(3, 2)
            
            risk = 0
            if hr > 100 or hr < 60: risk += 0.25
            if hr > 120 or hr < 50: risk += 0.25
            if sbp > 140 or sbp < 90: risk += 0.2
            if dbp > 90 or dbp < 60: risk += 0.15
            if spo2 < 95: risk += 0.3
            if spo2 < 90: risk += 0.3
            if temp > 99.5 or temp < 97.0: risk += 0.1
            if stress > 7: risk += 0.2
            
            risk = min(1.0, risk)
            if risk < 0.25:
                risk_label = 0
            elif risk < 0.6:
                risk_label = 1
            else:
                risk_label = 2
            
            data.append([hr, sbp, dbp, spo2, temp, stress, risk_label])
        
        return np.array(data)
    
    def train(self):
        """Train the advanced risk prediction model"""
        data = self.generate_synthetic_data()
        X = data[:, :6]
        y = data[:, 6]
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(" Health Risk Prediction Model Trained Successfully!")
    
    def predict_risk(self, heart_rate, systolic_bp, diastolic_bp, blood_oxygen, body_temperature=98.6, stress_level=3):
        """Predict health risk with advanced features"""
        if not self.is_trained:
            self.train()
        
        features = np.array([[heart_rate, systolic_bp, diastolic_bp, blood_oxygen, body_temperature, stress_level]])
        features_scaled = self.scaler.transform(features)
        
        risk_level = int(self.model.predict(features_scaled)[0])
        risk_proba = self.model.predict_proba(features_scaled)[0]
        
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_colors = ['success', 'warning', 'danger']
        
        return {
            'risk_level': risk_labels[risk_level],
            'risk_color': risk_colors[risk_level],
            'confidence': float(max(risk_proba)),
            'risk_score': float(risk_proba[risk_level])
        }
    
    def get_recommendations(self, metrics):
        """Generate personalized recommendations"""
        recommendations = []
        
        hr = metrics.get('heart_rate', 0)
        if hr > 100:
            recommendations.append("Your heart rate is elevated. Practice deep breathing and rest.")
        elif hr < 60:
            recommendations.append("Your heart rate is low. Monitor for dizziness or fatigue.")
        
        sbp = metrics.get('systolic_bp', 0)
        dbp = metrics.get('diastolic_bp', 0)
        if sbp > 140:
            recommendations.append("High systolic BP. Reduce salt intake and exercise regularly.")
        elif sbp < 90:
            recommendations.append("Low BP. Stay hydrated and avoid sudden position changes.")
        
        if dbp > 90:
            recommendations.append("High diastolic BP. Consult your doctor for hypertension screening.")
        
        spo2 = metrics.get('blood_oxygen', 100)
        if spo2 < 90:
            recommendations.append("CRITICAL: Low oxygen levels! Seek medical attention immediately!")
        elif spo2 < 95:
            recommendations.append("Low oxygen levels. Practice deep breathing exercises.")
        
        sleep = metrics.get('sleep_hours', 7)
        if sleep < 6:
            recommendations.append("You need more sleep. Aim for 7-8 hours for better health.")
        elif sleep > 9:
            recommendations.append("You're sleeping a lot. Consider consulting a doctor if persistent.")
        
        steps = metrics.get('steps', 0)
        if steps < 5000:
            recommendations.append("Increase your daily steps. Aim for 8000-10000 steps per day.")
        
        if not recommendations:
            recommendations.append("All vitals are within normal range! Keep up the healthy lifestyle!")
        
        return recommendations

# Initialize ML predictor
risk_predictor = HealthRiskPredictor()
risk_predictor.train()

# =============================================
# ALERT MANAGEMENT FUNCTIONS
# =============================================

def check_and_create_alerts(user_id, metrics):
    """Check health metrics and create alerts if needed"""
    alerts_created = []
    connection = get_db_connection()
    
    if not connection:
        return alerts_created
    
    cursor = connection.cursor()
    
    cursor.execute("SELECT alert_threshold_high_hr, alert_threshold_low_hr FROM users WHERE id = %s", (user_id,))
    thresholds = cursor.fetchone()
    
    high_hr = thresholds[0] if thresholds else 100
    low_hr = thresholds[1] if thresholds else 60
    
    hr = metrics.get('heart_rate', 0)
    if hr > high_hr:
        alerts_created.append(('warning','high', 'High Heart Rate', f"Heart rate is {hr} bpm above your threshold"))
    elif hr < low_hr:
        alerts_created.append(('warning','high', 'Low Heart Rate', f"Heart rate is {hr} bpm below your threshold"))
    
    sbp = metrics.get('systolic_bp', 0)
    if sbp > 140:
        alerts_created.append(('warning','high', 'High Blood Pressure', f"Systolic BP is {sbp} mmHg"))
    elif sbp < 90:
        alerts_created.append(('warning','medium', 'Low Blood Pressure', f"Systolic BP is {sbp} mmHg"))
    
    spo2 = metrics.get('blood_oxygen', 100)
    if spo2 < 90:
        alerts_created.append(('critical','critical', 'Critical Low Oxygen', f"Blood oxygen is {spo2}%"))
    elif spo2 < 95:
        alerts_created.append(('warning','high', 'Low Oxygen', f"Blood oxygen is {spo2}%"))
    
    for alert_category, severity, title, message in alerts_created:
        cursor.execute("""
            INSERT INTO alerts (user_id, alert_type, alert_category, severity, title, message, recommendation)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, 'vital', severity, title, message, "Please check your vitals and consult a doctor if needed"))
    
    connection.commit()
    cursor.close()
    connection.close()
    
    return alerts_created

# =============================================
# API ROUTES
# =============================================

@app.route('/api/register', methods=['POST'])
def register():
    """Advanced user registration - FIXED VERSION"""
    data = request.json
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    phone_number = data.get('phone_number')
    date_of_birth = data.get('date_of_birth')
    gender = data.get('gender')
    blood_group = data.get('blood_group')
    smoking_status = data.get('smoking_status', 'Never')
    alcohol_consumption = data.get('alcohol_consumption', 'Never')
    exercise_frequency = data.get('exercise_frequency', 'Moderate')
    
    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return jsonify({'error': 'Invalid email format'}), 400  
    
    password_hash = hash_password(password)
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor()
    
    try:
        age = None
        if date_of_birth:
            birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
            age = datetime.now().year - birth_date.year
        
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, first_name, last_name, 
                             phone_number, date_of_birth, age, gender, blood_group,
                             smoking_status, alcohol_consumption, exercise_frequency)  
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (username, email, password_hash, first_name, last_name, phone_number,
              date_of_birth, age, gender, blood_group, smoking_status, alcohol_consumption, exercise_frequency))
        
        user_id = cursor.lastrowid
        connection.commit()
        
        return jsonify({'message': 'User registered successfully', 'user_id': user_id}), 201
        
    except mysql.connector.IntegrityError as e:
        if 'username' in str(e):
            return jsonify({'error': 'Username already exists'}), 409
        elif 'email' in str(e):
            return jsonify({'error': 'Email already exists'}), 409
        return jsonify({'error': 'Registration failed'}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s AND is_active = 1", (username,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if user and verify_password(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user.get('role', 'user')
        
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("UPDATE users SET last_login_at = NOW() WHERE id = %s", (user['id'],))
            conn.commit()
            cur.close()
            conn.close()
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user['id'], 
                'username': user['username'], 
                'email': user['email'],
                'first_name': user.get('first_name'),
                'role': user.get('role', 'user')
            }
        }), 200
    
    else:
        return jsonify({'error': 'Invalid credentials or account inactive'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/health-metrics', methods=['POST'])
@login_required
def add_health_metrics():
    """Add advanced health metrics record"""
    data = request.json
    user_id = session['user_id']
    
    metrics = {
        'heart_rate': data.get('heart_rate'),
        'systolic_bp': data.get('systolic_bp'),
        'diastolic_bp': data.get('diastolic_bp'),
        'blood_oxygen': data.get('blood_oxygen', 98),
        'body_temperature': data.get('body_temperature', 98.6),
        'respiratory_rate': data.get('respiratory_rate', 16),
        'pain_level': data.get('pain_level', 0),
        'steps': data.get('steps', 0),
        'sleep_hours': data.get('sleep_hours', 7),
        'weight_kg': data.get('weight_kg'),
        'height_cm': data.get('height_cm'),
        'water_intake_ml': data.get('water_intake_ml', 0),
        'stress_level': data.get('stress_level', 3),
        'mood_score': data.get('mood_score', 7),
        'notes': data.get('notes', '')
    }
    
    if not all([metrics['heart_rate'], metrics['systolic_bp'], metrics['diastolic_bp']]):
        return jsonify({'error': 'Missing required health metrics'}), 400
    
    alerts = check_and_create_alerts(user_id, metrics)
    
    risk_prediction = risk_predictor.predict_risk(
        metrics['heart_rate'], metrics['systolic_bp'], metrics['diastolic_bp'],
        metrics['blood_oxygen'], metrics['body_temperature'], metrics['stress_level']
    )
    
    recommendations = risk_predictor.get_recommendations(metrics)
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor()
    
    bmi = None
    if metrics['weight_kg'] and metrics['height_cm']:
        bmi = metrics['weight_kg'] / ((metrics['height_cm'] / 100) ** 2)
    
    cursor.execute("""
        INSERT INTO health_metrics (
            user_id, heart_rate, systolic_bp, diastolic_bp, blood_oxygen,
            body_temperature, respiratory_rate, pain_level, steps, sleep_hours,
            weight_kg, height_cm, bmi, water_intake_ml, stress_level,
            mood_score, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        user_id, metrics['heart_rate'], metrics['systolic_bp'], metrics['diastolic_bp'],
        metrics['blood_oxygen'], metrics['body_temperature'], metrics['respiratory_rate'],
        metrics['pain_level'], metrics['steps'], metrics['sleep_hours'], metrics['weight_kg'],
        metrics['height_cm'], bmi, metrics['water_intake_ml'], metrics['stress_level'],
        metrics['mood_score'], metrics['notes']
    ))
    
    connection.commit()
    record_id = cursor.lastrowid
    cursor.close()
    connection.close()
    
    return jsonify({
        'message': 'Health metrics recorded successfully',
        'record_id': record_id,
        'alerts': alerts,
        'risk_assessment': risk_prediction,
        'recommendations': recommendations
    }), 201

@app.route('/api/health-metrics', methods=['GET'])
@login_required
def get_health_metrics():
    """Get user's health metrics history"""
    user_id = session['user_id']
    days = request.args.get('days', 30, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM health_metrics 
        WHERE user_id = %s AND recorded_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        ORDER BY recorded_at DESC
        LIMIT %s
    """, (user_id, days, limit))
    
    metrics = cursor.fetchall()
    cursor.close()
    connection.close()
    
    return jsonify({'metrics': metrics, 'count': len(metrics)}), 200

@app.route('/api/dashboard/stats', methods=['GET'])
@login_required
def get_dashboard_stats():
    """Get advanced dashboard statistics"""
    user_id = session['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM health_metrics 
        WHERE user_id = %s 
        ORDER BY recorded_at DESC 
        LIMIT 1
    """, (user_id,))
    latest = cursor.fetchone()
    
    cursor.execute("""
        SELECT 
            DATE(recorded_at) as date, 
            AVG(heart_rate) as avg_hr, 
            AVG(systolic_bp) as avg_sbp,
            AVG(diastolic_bp) as avg_dbp,
            AVG(blood_oxygen) as avg_o2,
            SUM(steps) as total_steps,
            AVG(sleep_hours) as avg_sleep,
            AVG(stress_level) as avg_stress
        FROM health_metrics 
        WHERE user_id = %s AND recorded_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY DATE(recorded_at)
        ORDER BY date ASC
    """, (user_id,))
    weekly_trend = cursor.fetchall()
    
    cursor.execute("""
        SELECT 
            severity, 
            alert_category,
            COUNT(*) as count 
        FROM alerts 
        WHERE user_id = %s AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY severity, alert_category
    """, (user_id,))
    alert_stats = cursor.fetchall()
    
    cursor.execute("""
        SELECT username, email, alert_threshold_high_hr, alert_threshold_low_hr,
               alert_threshold_high_bp_sys, alert_threshold_low_bp_sys
        FROM users WHERE id = %s
    """, (user_id,))
    user_info = cursor.fetchone()
    
    cursor.close()
    connection.close()
    
    risk_prediction = None
    if latest:
        risk_prediction = risk_predictor.predict_risk(
            latest['heart_rate'], latest['systolic_bp'], latest['diastolic_bp'],
            latest['blood_oxygen'], latest.get('body_temperature', 98.6), latest.get('stress_level', 3)
        )
    
    return jsonify({
        'latest_metrics': latest,
        'risk_assessment': risk_prediction,
        'weekly_trend': weekly_trend,
        'alert_stats': alert_stats,
        'user_info': user_info
    }), 200

@app.route('/api/alerts', methods=['GET'])
@login_required
def get_alerts():
    """Get user's alerts with filters"""
    user_id = session['user_id']
    unread_only = request.args.get('unread', 'false').lower() == 'true'
    severity = request.args.get('severity')
    limit = request.args.get('limit', 50, type=int)
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor(dictionary=True)
    
    query = "SELECT * FROM alerts WHERE user_id = %s"
    params = [user_id]
    
    if unread_only:
        query += " AND is_read = FALSE"
    
    if severity:
        query += " AND severity = %s"
        params.append(severity)
    
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    
    cursor.execute(query, params)
    alerts = cursor.fetchall()
    cursor.close()
    connection.close()
    
    return jsonify({'alerts': alerts, 'count': len(alerts)}), 200

@app.route('/api/alerts/<int:alert_id>/read', methods=['PUT'])
@login_required
def mark_alert_read(alert_id):
    """Mark alert as read"""
    user_id = session['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE alerts SET is_read = TRUE, read_at = NOW()
        WHERE id = %s AND user_id = %s
    """, (alert_id, user_id))
    connection.commit()
    cursor.close()
    connection.close()
    
    return jsonify({'message': 'Alert marked as read'}), 200

@app.route('/api/health-goals', methods=['POST'])
@login_required
def create_health_goal():
    """Create a new health goal"""
    data = request.json
    user_id = session['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO health_goals (user_id, goal_type, target_value, unit, start_date, target_date)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (user_id, data.get('goal_type'), data.get('target_value'), 
          data.get('unit'), data.get('start_date'), data.get('target_date')))
    
    connection.commit()
    goal_id = cursor.lastrowid
    cursor.close()
    connection.close()
    
    return jsonify({'message': 'Goal created successfully', 'goal_id': goal_id}), 201

@app.route('/api/health-goals', methods=['GET'])
@login_required
def get_health_goals():
    """Get user's health goals"""
    user_id = session['user_id']
    status = request.args.get('status', 'active')
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT * FROM health_goals 
        WHERE user_id = %s AND status = %s
        ORDER BY target_date ASC
    """, (user_id, status))
    goals = cursor.fetchall()
    cursor.close()
    connection.close()
    
    return jsonify({'goals': goals}), 200

@app.route('/api/reports', methods=['GET'])
@login_required
def generate_advanced_report():
    """Generate advanced health report"""
    user_id = session['user_id']
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    report_type = request.args.get('type', 'health')
    
    if not start_date or not end_date:
        return jsonify({'error': 'Start date and end date required'}), 400
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM health_metrics 
        WHERE user_id = %s AND DATE(recorded_at) BETWEEN %s AND %s
        ORDER BY recorded_at ASC
    """, (user_id, start_date, end_date))
    metrics = cursor.fetchall()
    
    cursor.execute("""
        SELECT username, email, first_name, last_name, age, gender, 
               blood_group, smoking_status, exercise_frequency
        FROM users WHERE id = %s
    """, (user_id,))
    user_info = cursor.fetchone()
    
    cursor.execute("""
        SELECT severity, COUNT(*) as count 
        FROM alerts 
        WHERE user_id = %s AND DATE(created_at) BETWEEN %s AND %s
        GROUP BY severity
    """, (user_id, start_date, end_date))
    alert_summary = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    if not metrics:
        return jsonify({'error': 'No data found for selected date range'}), 404
    
    hr_values = [m['heart_rate'] for m in metrics if m['heart_rate']]
    sbp_values = [m['systolic_bp'] for m in metrics if m['systolic_bp']]
    dbp_values = [m['diastolic_bp'] for m in metrics if m['diastolic_bp']]
    spo2_values = [m['blood_oxygen'] for m in metrics if m['blood_oxygen']]
    steps_values = [m['steps'] for m in metrics if m['steps']]
    sleep_values = [m['sleep_hours'] for m in metrics if m['sleep_hours']]
    
    report = {
        'user_info': user_info,
        'date_range': {'start': start_date, 'end': end_date},
        'summary': {
            'total_records': len(metrics),
            'avg_heart_rate': round(np.mean(hr_values), 1) if hr_values else 0,
            'avg_systolic_bp': round(np.mean(sbp_values), 1) if sbp_values else 0,
            'avg_diastolic_bp': round(np.mean(dbp_values), 1) if dbp_values else 0,
            'avg_blood_oxygen': round(np.mean(spo2_values), 1) if spo2_values else 0,
            'total_steps': sum(steps_values),
            'avg_sleep_hours': round(np.mean(sleep_values), 1) if sleep_values else 0,
            'alert_count': sum([a['count'] for a in alert_summary])
        },
        'alert_summary': alert_summary,
        'health_score': 0,
        'recommendations': []
    }
    
    return jsonify({'report': report}), 200

# =============================================
# FRONTEND ROUTES
# =============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/goals')
def goals():
    return render_template('goals.html')

@app.route('/batch-analyze')
def batch_analyze():
    return render_template('batch_analyze.html')

@app.route('/api/batch-ml-analyze', methods=['POST'])
@login_required
def batch_ml_analyze():
    """Batch ML Analysis for 500-1000 users with future prediction"""
    data = request.json.get('data', [])
    results = []
    
    for record in data:
        hr = record.get('heart_rate', 75) or 75
        sbp = record.get('systolic_bp', 120) or 120
        dbp = record.get('diastolic_bp', 80) or 80
        spo2 = record.get('blood_oxygen', 98) or 98
        age = record.get('age', 30) or 30
        stress = record.get('stress_level', 3) or 3
        sleep = record.get('sleep_hours', 7) or 7
        steps = record.get('steps', 7000) or 7000
        
        risk = risk_predictor.predict_risk(hr, sbp, dbp, spo2, 98.6, stress)
        
        future_risk_score = 0
        if hr > 95: future_risk_score += 0.3
        if sbp > 135: future_risk_score += 0.3
        if dbp > 85: future_risk_score += 0.2
        if spo2 < 96: future_risk_score += 0.2
        if stress > 6: future_risk_score += 0.2
        if sleep < 6.5: future_risk_score += 0.2
        if steps < 6000: future_risk_score += 0.2
        if age > 50: future_risk_score += 0.2
        
        future_risk = 'High' if future_risk_score > 0.5 else 'Low'
        
        health_score = 100
        if hr < 60 or hr > 100: health_score -= 15
        if sbp < 90 or sbp > 140: health_score -= 15
        if dbp < 60 or dbp > 90: health_score -= 10
        if spo2 < 95: health_score -= 20
        if stress > 7: health_score -= 10
        if sleep < 6: health_score -= 10
        if steps < 5000: health_score -= 10
        health_score = max(0, health_score)
        
        solutions = []
        if hr > 100: solutions.append("Practice deep breathing daily")
        if sbp > 140: solutions.append("Reduce salt intake, eat bananas")
        if dbp > 90: solutions.append("Consult doctor for BP checkup")
        if spo2 < 95: solutions.append("Practice breathing exercises")
        if stress > 7: solutions.append("Try meditation or yoga")
        if sleep < 6: solutions.append("Sleep 7-8 hours daily")
        if steps < 5000: solutions.append("Walk 30 minutes daily")
        
        if len(solutions) == 0:
            solutions.append("Continue healthy lifestyle")
        
        results.append({
            'heart_rate': hr,
            'systolic_bp': sbp,
            'diastolic_bp': dbp,
            'blood_oxygen': spo2,
            'risk_level': risk['risk_level'],
            'future_risk': future_risk,
            'health_score': health_score,
            'solutions': solutions
        })
    
    return jsonify(results), 200

# =============================================
# FEATURE 1: HEALTH TIME MACHINE (World's First)
# =============================================

@app.route('/api/time-machine/<int:user_id>', methods=['GET'])
@login_required
def health_time_machine(user_id):
    """Predicts past and future health trends - UNIQUE FEATURE"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # FIXED: Using recorded_at instead of created_at
        cursor.execute("""
            SELECT heart_rate, systolic_bp, diastolic_bp, blood_oxygen, 
                   health_score, DATE(recorded_at) as record_date
            FROM health_metrics 
            WHERE user_id = %s AND recorded_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            ORDER BY recorded_at ASC
        """, (user_id,))
        
        past_data = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if len(past_data) < 7:
            return jsonify({'success': False, 'error': 'Need at least 7 days of data'}), 400
        
        past_hr = [d['heart_rate'] for d in past_data[-30:] if d['heart_rate']]
        past_score = [d['health_score'] for d in past_data[-30:] if d['health_score']]
        
        def predict_future(values, days=30):
            if len(values) < 2:
                return [values[-1]] * days if values else [50] * days
            x = list(range(len(values)))
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(i*i for i in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
            intercept = (sum_y - slope * sum_x) / n
            
            future = []
            for i in range(n, n + days):
                pred = slope * i + intercept
                future.append(max(30, min(100, pred)))
            return future
        
        future_health_scores = predict_future(past_score, 30) if past_score else [70] * 30
        
        ideal_score = 85
        current_score = past_score[-1] if past_score else 70
        
        if future_health_scores[-1] > current_score:
            trend = "IMPROVING"
            message = "🎉 Good news! Your health is on an upward trajectory!"
        elif future_health_scores[-1] < current_score:
            trend = "DECLINING"
            message = "⚠️ Warning: Your health trend is declining. Action needed!"
        else:
            trend = "STABLE"
            message = "📊 Your health is stable. Maintain current habits."
        
        return jsonify({
            'success': True,
            'past_30_days_avg_hr': round(sum(past_hr) / len(past_hr), 1) if past_hr else 0,
            'current_health_score': current_score,
            'predicted_30_days_score': round(future_health_scores[-1], 1),
            'improvement_needed': max(0, ideal_score - current_score),
            'trend': trend,
            'time_machine_message': message,
            'future_predictions': future_health_scores[:7]
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================
# FEATURE 2: DISEASE WEATHER MAP (FIXED)
# =============================================

@app.route('/api/disease-weather/<int:user_id>', methods=['GET'])
@login_required
def disease_weather_map(user_id):
    """Predicts disease risk like weather forecast"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # FIXED: Using recorded_at instead of created_at
        cursor.execute("""
            SELECT heart_rate, systolic_bp, diastolic_bp, blood_oxygen, 
                   stress_level, sleep_hours, health_score
            FROM health_metrics 
            WHERE user_id = %s 
            ORDER BY recorded_at DESC LIMIT 14
        """, (user_id,))
        
        records = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if len(records) < 5:
            return jsonify({'success': False, 'error': 'Need at least 5 days of data'}), 400
        
        risk_score = 0
        for rec in records[:7]:
            day_risk = 0
            if rec['heart_rate'] and (rec['heart_rate'] > 100 or rec['heart_rate'] < 60):
                day_risk += 25
            if rec['systolic_bp'] and rec['systolic_bp'] > 140:
                day_risk += 30
            if rec['blood_oxygen'] and rec['blood_oxygen'] < 95:
                day_risk += 25
            if rec['stress_level'] and rec['stress_level'] > 7:
                day_risk += 15
            if rec['sleep_hours'] and rec['sleep_hours'] < 6:
                day_risk += 10
            risk_score += day_risk
        
        avg_risk = risk_score / min(7, len(records))
        
        if avg_risk > 70:
            weather = "🌪️ STORM WARNING - High disease risk! Take immediate action!"
            color = "red"
            action = "Rest today, consult doctor, avoid stress, drink more water"
        elif avg_risk > 40:
            weather = "☁️ CLOUDY - Moderate risk. Be careful!"
            color = "orange"
            action = "Take precautions, monitor closely, reduce salt intake"
        elif avg_risk > 15:
            weather = "⛅ PARTLY CLOUDY - Low risk. Stay aware!"
            color = "yellow"
            action = "Normal activities, but stay aware of your body"
        else:
            weather = "☀️ SUNNY - Very low risk! Enjoy your day!"
            color = "green"
            action = "Keep up the good work! Stay healthy"
        
        return jsonify({
            'success': True,
            'disease_weather': weather,
            'risk_percentage': round(avg_risk, 1),
            'action_required': action,
            'color': color
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================
# FEATURE 3: SILENT SYMPTOM DETECTOR (FIXED)
# =============================================

@app.route('/api/silent-symptoms/<int:user_id>', methods=['GET'])
@login_required
def silent_symptoms(user_id):
    """Detects symptoms that patient cannot express"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # FIXED: Using recorded_at instead of created_at
        cursor.execute("""
            SELECT heart_rate, systolic_bp, diastolic_bp, blood_oxygen,
                   stress_level, sleep_hours, steps, health_score,
                   DATE(recorded_at) as record_date
            FROM health_metrics 
            WHERE user_id = %s 
            ORDER BY recorded_at DESC LIMIT 30
        """, (user_id,))
        
        records = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if len(records) < 5:
            return jsonify({'success': False, 'error': 'Need more data'}), 400
        
        silent_symptoms = []
        
        hr_values = [r['heart_rate'] for r in records if r['heart_rate']]
        if hr_values and np.var(hr_values) > 100:
            silent_symptoms.append("🫀 Hidden Heart Rhythm Irregularity Detected")
        
        bp_values = [r['systolic_bp'] for r in records if r['systolic_bp']]
        if bp_values and np.var(bp_values) > 80:
            silent_symptoms.append("🩸 Silent Blood Pressure Fluctuations")
        
        sleep_issue = any(r['sleep_hours'] and r['sleep_hours'] < 6 for r in records[:5])
        stress_high = any(r['stress_level'] and r['stress_level'] > 7 for r in records[:5])
        if sleep_issue and stress_high:
            silent_symptoms.append("😴 Hidden Sleep-Stress Cycle Detected")
        
        o2_values = [r['blood_oxygen'] for r in records if r['blood_oxygen']]
        if o2_values and min(o2_values) < 94:
            silent_symptoms.append("💨 Occasional Oxygen Level Drops")
        
        if not silent_symptoms:
            silent_symptoms.append("✅ No silent symptoms detected. Good health!")
        
        return jsonify({
            'success': True,
            'silent_symptoms': silent_symptoms,
            'symptom_count': len([s for s in silent_symptoms if 'Hidden' in s or 'Silent' in s])
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================
# FEATURE 4: HEALTH GRAVITY SCORE (FIXED)
# =============================================

@app.route('/api/health-gravity/<int:user_id>', methods=['GET'])
@login_required
def health_gravity(user_id):
    """Measures health using gravity/black hole physics concept"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # FIXED: Using recorded_at instead of created_at
        cursor.execute("""
            SELECT heart_rate, systolic_bp, diastolic_bp, blood_oxygen,
                   bmi, stress_level, health_score
            FROM health_metrics 
            WHERE user_id = %s 
            ORDER BY recorded_at DESC LIMIT 1
        """, (user_id,))
        
        current = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not current:
            return jsonify({'success': False, 'error': 'No health data'}), 400
        
        gravity_score = 0
        
        bmi = current['bmi'] or 22
        if bmi > 30:
            gravity_score += 35
        elif bmi > 25:
            gravity_score += 20
        elif bmi < 18.5:
            gravity_score += 15
        
        hr = current['heart_rate'] or 75
        if hr > 100:
            gravity_score += 25
        elif hr > 90:
            gravity_score += 15
        
        sbp = current['systolic_bp'] or 120
        if sbp > 140:
            gravity_score += 30
        elif sbp > 130:
            gravity_score += 15
        
        o2 = current['blood_oxygen'] or 97
        if o2 < 90:
            gravity_score += 40
        elif o2 < 94:
            gravity_score += 20
        
        stress = current['stress_level'] or 3
        if stress > 7:
            gravity_score += 15
        
        if gravity_score >= 70:
            level = "🕳️ BLACK HOLE ZONE - Critical Gravity!"
            message = "Your health is collapsing. Immediate action required!"
        elif gravity_score >= 45:
            level = "🌍 HIGH GRAVITY - Heavy Health Load"
            message = "Body under significant stress. Need lifestyle changes."
        elif gravity_score >= 20:
            level = "🌙 NORMAL GRAVITY - Stable"
            message = "Normal health gravity. Maintain current routine."
        else:
            level = "🚀 LOW GRAVITY - Excellent!"
            message = "Your health is light and free! Keep it up!"
        
        return jsonify({
            'success': True,
            'gravity_score': gravity_score,
            'gravity_level': level,
            'escaping_velocity_needed': 100 - gravity_score,
            'message': message
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================
# FEATURE 5: RECOVERY TIME PREDICTOR (FIXED)
# =============================================

@app.route('/api/recovery-time/<int:user_id>', methods=['GET'])
@login_required
def recovery_time(user_id):
    """Predicts how many days to recover"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # FIXED: Using recorded_at instead of created_at
        cursor.execute("""
            SELECT heart_rate, systolic_bp, diastolic_bp, blood_oxygen,
                   stress_level, sleep_hours, health_score, recorded_at
            FROM health_metrics 
            WHERE user_id = %s 
            ORDER BY recorded_at DESC LIMIT 15
        """, (user_id,))
        
        records = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if len(records) < 3:
            return jsonify({'success': False, 'error': 'Need at least 3 records'}), 400
        
        current_score = records[0]['health_score'] or 70
        target_score = 85
        
        if current_score >= target_score:
            days = 0
            message = "✅ You're already in optimal health! No recovery needed."
        else:
            scores = [r['health_score'] for r in records if r['health_score']]
            if len(scores) > 1 and scores[-1] != scores[0]:
                improvement_rate = abs(scores[0] - scores[-1]) / len(scores) if len(scores) > 1 else 1
                if improvement_rate > 0:
                    days = int((target_score - current_score) / improvement_rate)
                    days = max(1, min(30, days))
                else:
                    days = 7
            else:
                days = 7
            
            if days <= 3:
                message = f"🎉 Quick recovery! About {days} days to optimal health."
            elif days <= 7:
                message = f"📈 Good progress! About {days} days to full recovery."
            else:
                message = f"⚠️ Recovery will take about {days} days. Be patient and consistent."
        
        return jsonify({
            'success': True,
            'current_score': current_score,
            'target_score': target_score,
            'estimated_recovery_days': days,
            'message': message,
            'score_gap': max(0, target_score - current_score)
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================
# MAIN ENTRY POINT
# =============================================

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 ADVANCED MACHINE LEARNING HEALTH MONITORING SYSTEM")
    print("=" * 50)
    print("✅ Starting Flask Application...")
    print("📊 Database: health_monitoring")
    print("🤖 ML Model: Random Forest (Advanced)")
    print("🌐 Server: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)