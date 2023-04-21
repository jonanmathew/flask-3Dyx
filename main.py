# ----------------------------- App imports -----------------------------
import os
from datetime import datetime
from flask import Flask, jsonify, request
from pymongo import MongoClient
from firebase import auth
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS

# ----------------------------- App config -----------------------------
app = Flask(__name__)

load_dotenv()

# ----------------------------- CORS -----------------------------
client_url = os.getenv("CLIENT_URL")
CORS(app, origins=client_url.split('|'), supports_credentials=True)

# ----------------------------- DB config -----------------------------
mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)
message_db = client.classio_message_db
message_collection = message_db.messages
main_db = client.classio_main_db
student_collection = main_db.students
teacher_collection = main_db.teachers


# ----------------------------- Auth verification middleware -----------------------------


def auth_verify_token(func):
    @wraps(func)
    def middleware(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token:
            return jsonify({'error': 'Missing token'}), 401
        if id_token.startswith("Bearer "):
            id_token = id_token[len("Bearer "):]
        decoded_token = auth.verify_id_token(id_token)
        if not id_token:
            return jsonify({'error': 'User Verification Failed'}), 401
        user_id = decoded_token['user_id']
        name = decoded_token['name']
        user_type = name.split("|")[0]
        user_code = name.split("|")[1]
        request.user_code = user_code
        request.user_id = user_id
        request.user_type = user_type
        return func(*args, **kwargs)
    return middleware

# ----------------------------- Other Functions -----------------------------


def get_sender_id():
    user_id = request.user_id
    user_type = request.user_type
    sender_id = ''
    if user_type == 'student':
        sender_id = student_collection.find_one({'uid': user_id}, {'_id': 1})
    elif user_type == 'teacher':
        sender_id = teacher_collection.find_one({'uid': user_id}, {'_id': 1})
    return str(sender_id['_id'])

# ----------------------------- Routes and Controllers -----------------------------


@app.route('/api/get-users', methods=['GET'])
@auth_verify_token
def get_users():
    user_code = request.user_code
    students = student_collection.find({'code': user_code}, {"uid": 1})
    teachers = teacher_collection.find({'code': user_code}, {"uid": 1})
    users = []
    for student in list(students):
        user = auth.get_user(student["uid"])
        users.append({"userId": str(student["_id"]), "name": user.display_name.split(
            "|")[2], "photoURL": user.photo_url})
    for teacher in list(teachers):
        user = auth.get_user(teacher["uid"])
        users.append({"userId": str(teacher["_id"]), "name": user.display_name.split(
            "|")[2], "photoURL": user.photo_url})
    return jsonify({'users': users})


@app.route("/api/send-message", methods=["POST"])
@auth_verify_token
def send_message():
    sender_id = get_sender_id()
    receiver_id = request.args.get('receiverId')
    data = request.json
    message = data.get('message')
    created_at = datetime.utcnow()
    message_doc = {
        'senderId': sender_id,
        'receiverId': receiver_id,
        'message': message,
        'createdAt': created_at,
        'read': False
    }
    message_collection.insert_one(message_doc)
    return jsonify({'sentMessage': True})


@app.route('/api/get-chats', methods=['GET'])
@auth_verify_token
def get_chats():
    sender_id = get_sender_id()
    messagess = message_collection.find({'senderId': sender_id}, {"_id": 0})
    receiver_ids = []
    for msg in messagess:
        receiver_id = msg['receiverId']
        receiver_ids.append(receiver_id)
    final_receiver_ids = list(set(receiver_ids))
    created_at_arr = []
    unread_arr = []
    for ri in final_receiver_ids:
        mssgss = message_collection.find(
            {'senderId': sender_id, 'receiverId': ri}, {"_id": 0})
        messages_list = list(mssgss)
        created_at_arr.append(messages_list[-1]["createdAt"])
        ctr = 0
        for mssg in messages_list:
            if (mssg['read'] == False):
                ctr = 1
                break
        if (ctr == 1):
            unread_arr.append(True)
        else:
            unread_arr.append(False)
    final_arr = []
    for i, ri in enumerate(final_receiver_ids):
        final_arr.append({
            "userId": ri,
            "unread": unread_arr[i],
            "createdAt": created_at_arr[i]
        })
    final_arr_sorted = sorted(final_arr, key=lambda user: user["createdAt"])
    return jsonify({'chats': final_arr_sorted})


@app.route('/api/get-messages', methods=['GET'])
@auth_verify_token
def get_messages():
    sender_id = get_sender_id()
    receiver_id = request.args.get('receiverId')
    mssgs = message_collection.find(
        {'senderId': sender_id, 'receiverId': receiver_id}, {"_id": 0})
    message_list = []
    for msg in list(mssgs):
        message_list.append({
            'message': msg['message'],
            'createdAt': msg['createdAt'],
            "read": msg['read']
        })
    message_collection.update_many(
        {'senderId': sender_id, 'receiverId': receiver_id, 'read': False}, {"$set": {'read': True}})
    return jsonify({'messages': message_list})


@app.route('/api/get-new-messages', methods=['GET'])
@auth_verify_token
def get_new_messages():
    sender_id = get_sender_id()
    receiver_id = request.args.get('receiverId')
    mssgs = message_collection.find(
        {'senderId': sender_id, 'receiverId': receiver_id, 'read': False}, {"_id": 0})
    message_list = []
    for msg in list(mssgs):
        message_list.append({
            'message': msg['message'],
            'createdAt': msg['createdAt'],
        })
    message_collection.update_many(
        {'senderId': sender_id, 'receiverId': receiver_id, 'read': False}, {"$set": {'read': True}})
    return jsonify({'new_messages': message_list})


@app.route("/api/send-message-chatbot", methods=["POST"])
@auth_verify_token
def send_message_chatbot():
    sender_id = get_sender_id()
    data = request.json
    message = data.get('message')
    created_at = datetime.utcnow()
    message_doc = {
        'senderId': sender_id,
        'receiverId': "69",
        'message': message,
        'createdAt': created_at,
        'read': True
    }
    message_collection.insert_one(message_doc)
    reply_doc = {
        'senderId': "69",
        'receiverId': sender_id,
        'message': message,
        'createdAt': created_at,
        'read': False
    }
    return jsonify({'sentMessage': True, 'replyMessage': reply_doc})


# ----------------------------- Application run -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
