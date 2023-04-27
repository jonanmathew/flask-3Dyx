# ----------------------------- App imports -----------------------------
import os
from datetime import datetime
from flask import Flask, jsonify, request
from pymongo import MongoClient
from firebase import auth
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin

# ----------------------------- App config -----------------------------
app = Flask(__name__)

load_dotenv()

# ----------------------------- CORS -----------------------------
client_url = os.getenv("CLIENT_URL")
CORS(app, origins=client_url, supports_credentials=True)

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
@cross_origin(supports_credentials=True)
@auth_verify_token
def get_users():
    sender_id = get_sender_id()
    user_code = request.user_code
    students = student_collection.find({'code': user_code}, {"uid": 1})
    teachers = teacher_collection.find({'code': user_code}, {"uid": 1})
    users = []
    for student in list(students):
        user = auth.get_user(student["uid"])
        name = user.display_name.split(
            "|")[2]
        if str(student['_id']) != sender_id:
            users.append(
                {"userId": str(student["_id"]), 'email': user.email, "name": name, "photoURL": user.photo_url})

    for teacher in list(teachers):
        user = auth.get_user(teacher["uid"])
        name = user.display_name.split(
            "|")[2]
        if str(teacher['_id']) != sender_id:
            users.append(
                {"userId": str(teacher["_id"]), 'email': user.email, "name": name, "photoURL": user.photo_url})

    return jsonify({'users': users})


@app.route('/api/get-chats', methods=['GET'])
@cross_origin(supports_credentials=True)
@auth_verify_token
def get_chats():
    sender_id = get_sender_id()
    messages = message_collection.find(
        {'$or': [{'senderId': sender_id}, {'receiverId': sender_id}]}, {"_id": 0})
    receiver_ids = []
    for msg in list(messages):
        msg_receiver_id = msg['receiverId']
        msg_sender_id = msg['senderId']
        if msg_receiver_id == sender_id:
            receiver_ids.append(msg_sender_id)
        elif msg_sender_id == sender_id:
            receiver_ids.append(msg_receiver_id)
    final_receiver_ids = list(set(receiver_ids))
    last_message = []
    receiver_last_message = []
    unread_arr = []
    for receiver_id in final_receiver_ids:
        mssgs = message_collection.find(
            {'senderId': {'$in': [sender_id, receiver_id]}, 'receiverId': {'$in': [sender_id, receiver_id]}}, {"_id": 0})
        messages_list = list(mssgs)
        messages_list.reverse()
        last_message.append(messages_list[0]["createdAt"])
        is_unread = False
        is_already_added = False
        for mssg in messages_list:
            if mssg['receiverId'] == sender_id:
                if not is_already_added:
                    print(mssg['receiverId'], sender_id)
                    receiver_last_message.append(mssg["createdAt"])
                if mssg['read'] == False:
                    is_unread = True
                if is_already_added and is_unread:
                    break
        if not is_already_added:
            receiver_last_message.append('')
        if (is_unread):
            unread_arr.append(True)
        else:
            unread_arr.append(False)
    chats = []
    for i, rid in enumerate(final_receiver_ids):
        chats.append({
            "userId": rid,
            "unread": unread_arr[i],
            'lastMessage': last_message[i],
            "receiverLastMessage": receiver_last_message[i]
        })
    chats_sorted = sorted(chats, key=lambda user: datetime.strptime(
        user["lastMessage"].strftime('%Y-%m-%d %H:%M:%S'), "%Y-%m-%d %H:%M:%S"), reverse=True)
    return jsonify({'chats': chats_sorted})


@app.route("/api/send-message", methods=["POST"])
@cross_origin(supports_credentials=True)
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


@app.route('/api/get-messages', methods=['GET'])
@cross_origin(supports_credentials=True)
@auth_verify_token
def get_messages():
    sender_id = get_sender_id()
    receiver_id = request.args.get('receiverId')
    mssgs = message_collection.find(
        {'senderId': {'$in': [sender_id, receiver_id]}, 'receiverId': {'$in': [sender_id, receiver_id]}}, {"_id": 0})
    message_list = []
    for msg in list(mssgs):
        sender = False
        if msg['senderId'] == sender_id:
            sender = True
        message_list.append({
            'sender': sender,
            'message': msg['message'],
            'createdAt': msg['createdAt'],
            "read": msg['read']
        })
    message_collection.update_many(
        {'senderId': receiver_id, 'receiverId': sender_id, 'read': False}, {"$set": {'read': True}})
    return jsonify({'messages': message_list})


@app.route('/api/get-new-messages', methods=['GET'])
@cross_origin(supports_credentials=True)
@auth_verify_token
def get_new_messages():
    sender_id = get_sender_id()
    receiver_id = request.args.get('receiverId')
    mssgs = message_collection.find(
        {'senderId': receiver_id, 'receiverId': sender_id, 'read': False}, {"_id": 0})
    message_list = []
    for msg in list(mssgs):
        message_list.append({
            'sender': False,
            'message': msg['message'],
            'createdAt': msg['createdAt'],
            'read': True
        })
    message_collection.update_many(
        {'senderId': receiver_id, 'receiverId': sender_id, 'read': False}, {"$set": {'read': True}})
    return jsonify({'newMessages': message_list})


@app.route("/api/send-message-chatbot", methods=["POST"])
@cross_origin(supports_credentials=True)
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
