from flask import Flask, jsonify, request
import os
import datetime
from pymongo import MongoClient

app = Flask(__name__)

mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)
db = client.classio_message_db
collection = db.messages

@app.route("/add-message", methods=["POST"])
def store_message():
    data = request.json

    sender_id = data.get('senderId')
    receiver_id = data.get('receiverId')
    message = data.get('message')
    created_at = datetime.datetime.utcnow().isoformat()
    
    message_doc = {
        'senderId': sender_id,
        'receiverId': receiver_id,
        'message': message,
        'createdAt':created_at,
        'read': False
        
    }
    
    collection.insert_one(message_doc)
    return jsonify({'addedMessage': True})
    
@app.route('/get-users', methods=['GET'])
def get_users():

    sender_id = request.args.get('senderId')
    messagess = collection.find({'senderId': sender_id},{"_id":0})
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
        mssgss = collection.find({'senderId': sender_id,'receiverId': ri},{"_id":0})
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
                    
        
@app.route('/get-messages', methods=['GET'])
def get_messages():

    sender_id = request.args.get('senderId')
    receiver_id = request.args.get('receiverId')
    m = collection.find({'senderId': sender_id,'receiverId': receiver_id},{"_id":0})
    collection.update_many({'senderId': sender_id,'receiverId': receiver_id,'read':False},{"$set":{'read':True}})
    message_list = []
    for msg in m:
        message_list.append({
            'message': msg['message'],
            'createdAt': msg['createdAt']
        })
    return jsonify({'messages': message_list})

@app.route('/get-new-messages', methods=['GET'])
def get_new_messages():

    sender_id = request.args.get('senderId')
    receiver_id = request.args.get('receiverId')
    m = collection.find({'senderId': sender_id,'receiverId': receiver_id,'read':False},{"_id":0})
    collection.update_many({'senderId': sender_id,'receiverId': receiver_id,'read':False},{"$set":{'read':True}})
    message_list = []
    for msg in m:
        message_list.append({
            'message': msg['message'],
            'createdAt': msg['createdAt']
        })
    return jsonify({'messages': message_list})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
