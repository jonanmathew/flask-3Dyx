from flask import Flask, jsonify, request
import os
import datetime
from pymongo import MongoClient
from firebase import auth
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=['http://localhost:3000', 'https://classio.vercel.app'], supports_credentials=True)

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)
message_db = client.classio_message_db
message_collection = message_db.messages
main_db= client.classio_main_db
student_collection = main_db.students
teacher_collection = main_db.teachers

def auth_verify_token(func):
    @wraps(func)
    def middleware(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token:
            return jsonify({'error': 'Missing token'}), 401
        decoded_token = auth.verify_id_token(id_token)
        if not id_token:
            return jsonify({'error': 'User Verification Failed'}), 401
        uid = decoded_token['user_id']
        name=decoded_token['name']
        code=name.split("|")[1]
        user_type=name.split("|")[0]
        request.code=code
        request.uid=uid
        request.user_type=user_type
        return func(*args, **kwargs)

    return middleware

@app.route('/api/get-users', methods=['GET'])
@auth_verify_token
def get_users():
    code = request.code  
    students = student_collection.find({'code': code},{"uid":1})
    teachers = teacher_collection.find({'code': code},{"uid":1})
    users=[]
    for student in list(students):
        user=auth.get_user(student["uid"])
        users.append({"userID":str(student["_id"]),"displayName":user.display_name.split("|")[2],"photoURL":user.photo_url})
    for teacher in list(teachers):
        user=auth.get_user(teacher["uid"])
        users.append({"userID":str(teacher["_id"]),"displayName":user.display_name.split("|")[2],"photoURL":user.photo_url})
    return jsonify({'users': users})

    
@app.route("/api/add-message", methods=["POST"])
@auth_verify_token
def store_message():
    data = request.json
    sender_id = request.args.get('senderId')  
    receiver_id = request.args.get('receiverId')
    message = data.get('message')
    created_at = datetime.datetime.utcnow().isoformat()    
    message_doc = {
        'senderId': sender_id,
        'receiverId': receiver_id,
        'message': message,
        'createdAt':created_at,
        'read': False    
    }
    message_collection.insert_one(message_doc)
    return jsonify({'addedMessage': True})
    
@app.route('/api/get-chats', methods=['GET'])
@auth_verify_token
def get_chats():
    sender_id = request.args.get('senderId')
    messagess = message_collection.find({'senderId': sender_id},{"_id":0})
    receiver_ids=[]
    for msg in messagess:
        receiver_id = msg['receiverId']
        created_at = msg['createdAt']
        read = msg['read']
        receiver_ids.append(receiver_id)
    final_receiver_ids = list(set(receiver_ids))
    created_at_arr=[]
    unseen_arr=[]
    for ri in final_receiver_ids:
        mssgss = message_collection.find({'senderId': sender_id,'receiverId': ri},{"_id":0})
        messages_list=list(mssgss)
        created_at_arr.append(messages_list[-1]["createdAt"])
        ctr=0
        for mssg in mssgss:
            if(mssg['read']==False):
                ctr=1
                break
        if(ctr==1):
            unseen_arr.append(True)
        else:
            unseen_arr.append(False)        
    final_arr=[]       
    for i,ri in enumerate(final_receiver_ids):
        final_arr.append({
            "id":ri,
            "unseen":unseen_arr[i],
            "createdAt":created_at_arr[i]
        })
    print(final_arr)
    final_arr_sorted=sorted(final_arr,key=lambda user:user["createdAt"])      
    return jsonify({'users': final_arr_sorted})
                    
        
@app.route('/api/get-messages', methods=['GET'])
@auth_verify_token
def get_messages():
    sender_id = request.args.get('senderId')
    receiver_id = request.args.get('receiverId')
    m = message_collection.find({'senderId': sender_id,'receiverId': receiver_id},{"_id":0})
    message_collection.update_many({'senderId': sender_id,'receiverId': receiver_id,'read':False},{"$set":{'read':True}})
    message_list = []
    for msg in m:
        message_list.append({
            'message': msg['message'],
            'createdAt': msg['createdAt']
        })
    return jsonify({'messages': message_list})

@app.route('/api/get-new-messages', methods=['GET'])
@auth_verify_token
def get_new_messages():
    sender_id = request.args.get('senderId')
    receiver_id = request.args.get('receiverId')
    m = message_collection.find({'senderId': sender_id,'receiverId': receiver_id,'read':False},{"_id":0})
    message_collection.update_many({'senderId': sender_id,'receiverId': receiver_id,'read':False},{"$set":{'read':True}})
    message_list = []
    for msg in m:
        print(msg)
        message_list.append({
            'message': msg['message'],
            'createdAt': msg['createdAt']
        })
    return jsonify({'messages': message_list})

@app.route("/api/add-chatbot-message", methods=["POST"])
@auth_verify_token
def chatbot_message():
    data = request.json
    sender_id = data.get('senderId')
    message = data.get('message')
    created_at = datetime.datetime.utcnow().isoformat()
    message_doc = {
        'senderId': sender_id,
        'receiverId': "69",
        'message': message,
        'createdAt':created_at,
        'read': False
        
    }
    message_collection.insert_one(message_doc)
    reply_doc = {
        'senderId': "69",
        'receiverId': sender_id,
        'message': message,
        'createdAt':created_at,
        'read': False
    }
    return jsonify({'Message': reply_doc})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
