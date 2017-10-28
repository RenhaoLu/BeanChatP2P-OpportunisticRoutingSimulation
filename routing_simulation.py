#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math
import re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#Simulation Parameters
SIMULATION_RUNTIME = 5000
#SIMULATION_RUNTIME = 500
STATIC_NODE_COUNT = 10
MOBILE_USER_COUNT = 100

#Run deterministically for reproducability purposes
random.seed(0)

class MobileUser:
	def __init__(self, my_name, mm, start_place):
		self.name = my_name
		self.markov_model = mm
		self.current_location = start_place
		self.stored_models = {}
		self.tx_queue = []
		self.my_messages = []
	
	def step(self, ts):
		#First record any messages for which the user is the intended recipient
		for m in self.tx_queue:
			if m['dst'] == self.name:
				self.my_messages.append({'message' : m['message'], 'src' : m['src'], 'dst' : m['dst'] ,'timestamp' : ts})
				del self.tx_queue[self.tx_queue.index(m)]
		
		#Get the relevant section of the Markov model
		next_probabilities = self.markov_model[self.current_location]
		#Probabilistic transition
		choice_list = []
		for key in next_probabilities:
			tmp = [key] * int(10 * next_probabilities[key])
			choice_list = choice_list + tmp
		
		future_location = random.choice(choice_list)
		at_new_location = bool(self.current_location == future_location)
		self.current_location = future_location
		if at_new_location:
			return future_location
		else:
			return None
	
	def append_tx_queue(self, packet):
		self.tx_queue.append(packet)
	
	def append_opportunistic_tx_queue(self, pkt):
		#Remove the first item from the routing path, then store
		mod_pkt = pkt.copy()
		mod_route = mod_pkt['routing_path']
		del mod_route[0]
		mod_pkt['routing_path'] = mod_route
		#Store mod_pkt
		self.tx_queue.append(mod_pkt)
	
	#If any messages are available for node rx, pop them off.
	def dequeue_for(self, rx):
		deletes = []
		for pkt_index in range(0, len(self.tx_queue)):
			pkt = self.tx_queue[pkt_index]
			pkt_route = pkt['routing_path']
			if len(pkt_route) < 1:
				continue
			if pkt_route[0] == rx:
				deletes.append(pkt)
				yield pkt
		
		for d in deletes:
			del self.tx_queue[self.tx_queue.index(d)]
	
	def get_opportunistic_route(self, current_loc, rx_user):
		#Build a NetworkX model of the known opportunistic connections
		net_graph = nx.Graph()
		for other_node in self.stored_models:
			for other_user in self.stored_models[other_node]:
				net_graph.add_edge(other_node, other_user, hit_count=self.stored_models[other_node][other_user])
				#print "Adding edge: " + str(other_node) + ", " + str(other_user) + " = " + str(self.stored_models[other_node][other_user])
		
		#Get largest hit_count
		hits = []
		graph_edges = net_graph.edges()
		for ge in graph_edges:
			src = ge[0]
			tgt = ge[1]
			edge_data = net_graph.get_edge_data(src, tgt)
			#print edge_data
			if edge_data == None:
				continue
			if not 'hit_count' in edge_data:
				continue
			hit_count = edge_data['hit_count']
			hits.append(hit_count)
		
		largest_hit = max(hits)
		
		
		#Calculate probability as this(hit_count) / max(hit_count) and edge weight as -log(prob)
		for ge in graph_edges:
			src = ge[0]
			tgt = ge[1]
			edge_data = net_graph.get_edge_data(src, tgt)
			if edge_data == None:
				continue
			if not 'hit_count' in edge_data:
				continue
			hit_count = edge_data['hit_count']
			this_prob = float(hit_count) / float(largest_hit)
			this_weight = -math.log10(this_prob)
			net_graph.add_edge(src, tgt, weight=this_weight)
		
		#Now use NetworkX method for finding the lowest cost path
		return nx.shortest_path(net_graph, source=current_loc, target=rx_user, weight='weight')
					

class StaticNode:
	def __init__(self):
		self.message_queue = []
		
		"""
		Keep track of nodes which this static node is likely to come
		into direct contact with.
		"""
		self.direct_probabilistic_model = {}
		
		"""
		Keep track of the direct probabilistic models of other nodes
		"""
		self.other_nodes = {}
	
	def update_list(self, other_node, other_node_list):
		self.other_nodes[other_node] = other_node_list
	
	def check_in(self, username):
		if username in self.direct_probabilistic_model:
			self.direct_probabilistic_model[username] = self.direct_probabilistic_model[username] + 1
		else:
			self.direct_probabilistic_model[username] = 1
	
	def enqueue_opportunistic_message(self, pkt):
		#Remove the first item from the routing path, then store
		mod_pkt = pkt.copy()
		mod_route = mod_pkt['routing_path']
		del mod_route[0]
		mod_pkt['routing_path'] = mod_route
		#Store mod_pkt
		self.message_queue.append(mod_pkt)
		
	#If any messages are available for mobile node rx, pop them off.
	def dequeue_for(self, rx):
		deletes = []
		for pkt_index in range(0, len(self.message_queue)):
			pkt = self.message_queue[pkt_index]
			pkt_route = pkt['routing_path']
			if pkt_route[0] == rx:
				deletes.append(pkt)
				yield pkt
		
		for d in deletes:
			del self.message_queue[self.message_queue.index(d)]



def generate_random_behaviour(places_list):
	markov_model = {}
	for current_place in places_list:
		transition_probabilities = {}
		#Where N is the number of places in places_list...
		#...find N numbers between 0.0 to 1.0 inclusive that sum to 1
		random_n_nums = []
		for next_place in places_list:
			random_n_nums.append(random.random())
		
		random_n_nums = np.array(random_n_nums)
		random_n_nums = random_n_nums / sum(random_n_nums) #Normalize
		random_n_nums = np.round(10 * random_n_nums)/10 #To 1 decimal place
		index = 0
		for next_place in places_list:
			transition_probabilities[next_place] = random_n_nums[index]
			index += 1
		
		markov_model[current_place] = transition_probabilities
	
	return markov_model



#Randomly generate STATIC_NODE_COUNT number of StaticNodes
static_nodes = {}
for si in range(STATIC_NODE_COUNT):
	static_nodes["StaticNode_" + str(si)] = StaticNode()

#Randomly generate MOBILE_USER_COUNT number of MobileUsers
mobile_users = {}
for mi in range(MOBILE_USER_COUNT):
	#Generate the behaviour model
	this_behaviour = generate_random_behaviour(static_nodes.keys())
	#Random start location
	this_start = random.choice(static_nodes.keys())
	#Generate the user
	mobile_users["MobileUser_" + str(mi)] = MobileUser("MobileUser_" + str(mi), this_behaviour, this_start)




#Run the simulation...

#Keep track of all messages; timeframe sent and timeframe received
messages_sent = {}

for usr in mobile_users:
	messages_sent[usr] = 0

messages_for = {}

for usr in mobile_users:
	messages_for[usr] = []

#Reliability of sucessfully sending and receiving a message at the given timeframe
quality_tracking = []

for timeframe in range(SIMULATION_RUNTIME):
	#Step all MobileUsers
	for usr in mobile_users:
		###
		this_user = mobile_users[usr]
		new_location = this_user.step(timeframe)
		
		#If user has moved between StaticNodes...
		if new_location != None:
			#check them into new static node
			static_nodes[new_location].check_in(usr)
			
			#also, synchronize...
			#...first, download direct_probabilistic_model from node
			new_location_direct_probabilistic_model = static_nodes[new_location].direct_probabilistic_model.copy()
			
			#...then push all downloaded models to this static node
			for model in this_user.stored_models:
				static_nodes[new_location].update_list(model, this_user.stored_models[model])
			
			#...finally update user's stored model list
			this_user.stored_models[new_location] = new_location_direct_probabilistic_model.copy()
		
		#Either way, we are at at static node so if any opportunistic message need to be sent to it, do it.
		for opp_msg in this_user.dequeue_for(new_location):
			static_nodes[new_location].enqueue_opportunistic_message(opp_msg)

		#Are there any opportunistic messages stored at the current static node for user to take
		for opp_msg in static_nodes[this_user.current_location].dequeue_for(usr):
			this_user.append_opportunistic_tx_queue(opp_msg)
		
		
		#Should this user send a message or not p(yes)=0.1, p(no)=0.9
		should_send = bool(random.uniform(0,1) < 0.1)
		if should_send:
			#Decide whom to send the message to
			possible_rx = mobile_users.keys()
			del possible_rx[possible_rx.index(usr)]
			#Equal probability to send a message to any user in possible_rx
			my_rx = random.choice(possible_rx)
			
			#Let the performance tracker know that the user is sending a message
			messages_sent[usr] = messages_sent[usr] + 1
			
			msg_payload = "Message sent at TIMEFRAME=" + str(timeframe)
			msg_src = str(usr)
			msg_dst = str(my_rx)
			messages_for[my_rx].append({'message' : msg_payload, 'src' : msg_src, 'dst' : msg_dst})
			
			#...simulate message sending
			#..... first get opportunistic route
			try:
				msg_route = this_user.get_opportunistic_route(this_user.current_location, my_rx)
				#.......... now assemble the message packet
				message_packet = {}
				message_packet['message'] = msg_payload
				message_packet['src'] = msg_src
				message_packet['dst'] = msg_dst
				message_packet['routing_path'] = msg_route
				#Store this packet in our TX queue
				this_user.append_tx_queue(message_packet)
			except:
				#print "Insufficient network knowledge for " + str(usr)
				pass
	
	#Get instantaneous link quality
	inst_total_tx_messages = 0
	inst_total_rx_messages = 0
	for usr in messages_for:
		for msg in messages_for[usr]:
			inst_total_tx_messages += 1
			for user_msg in mobile_users[usr].my_messages:
				#print msg
				#print user_msg
				if msg['message'] == user_msg['message']:
					if msg['src'] == user_msg['src']:
						if msg['dst'] == user_msg['dst']:
							inst_total_rx_messages += 1
	
	#Calculate it...
	try:
		inst_link_qual = 100 * float(inst_total_rx_messages) / float(inst_total_tx_messages)
	except ZeroDivisionError:
		inst_link_qual = 0
		
	quality_tracking.append(inst_link_qual)		
	
	print "Timeframe: " + str(timeframe)
	


#Simulation Results
print "### Simulation Results ###"
for mu in mobile_users:
	print "*** " + str(mu) + " ***"
	print "Messages sent: " + str(messages_sent[mu])
	print "Mailbox: " + str(mobile_users[mu].my_messages)
	print "TX: Queue: " + str(mobile_users[mu].tx_queue)
	print "---------------------------------"


print ""
print ""
print ""

#Calculate performance results
total_messages_sent = 0
for user in messages_sent:
	total_messages_sent += messages_sent[user]


total_messages_received = 0
for user in mobile_users:
	total_messages_received += len(mobile_users[user].my_messages)

message_sucess_prob = 100 * float(total_messages_received) / float(total_messages_sent)

#Performance result metric: message delay
delays = []
timestamp_regex = re.compile('.+ TIMEFRAME=(\d+)')
#...go through all users
for usr in mobile_users:
	#...go through all their messages
	for msg in mobile_users[usr].my_messages:
		#...get the received timestamp
		timestamp_rx = int(msg['timestamp'])
		#...get the transmitted timestamp
		#...parse the message
		msg_payload = msg['message']
		msg_match = timestamp_regex.match(msg_payload)
		timestamp_tx = int(msg_match.group(1))
		delay = timestamp_rx - timestamp_tx
		delays.append(delay)
		

print "### Performance Results ###"
print "Total messages received: " + str(total_messages_received)
print "Total messages sent: " + str(total_messages_sent)
print "Message sucess probability: " + str(message_sucess_prob) + "%"

#Graph the delays
delay_domain = range(0,max(delays) + 1)
delay_range = []
sumtest = 0
for dly in delay_domain:
	delay_range.append(float(delays.count(dly)) / float(total_messages_sent))
	sumtest += float(delays.count(dly)) / float(total_messages_sent)

print "SUMTEST={}".format(sumtest)

plt.subplot(211)
plt.title("Message Delay Probability Distribution")
plt.ylabel("Probability")
plt.xlabel("Delay (time frames)")
plt.stem(delay_domain, delay_range)

plt.subplot(212)
plt.title("Opportunistic Network Discovery")
plt.ylabel("Probabilty of Sucessful Message Transmission (%)")
plt.xlabel("Simulation (time frame)")
plt.plot(range(0,SIMULATION_RUNTIME), quality_tracking)

plt.show()
