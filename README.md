# ElevatorSaga

## Reinforcement learning control agent for Elevators, focusing on minimizing average wait time and maximizing throuput in a large stochastic enviorment

By: Eran Nussinovitch, Gregory Pasternak, Asaph Shamir

## Introduction
Elevators are one of the most used means of transport for
both people and goods. Elevators were initially operated manually
by a dedicated person which was always present in the
elevator. Today, elevators are operated by a controller, which
given the pressed buttons in the elevator and the different
floors uses some algorithm to decide where the elevator will
go. A particularly bad example of such mechanism can be
found in the Rothberg B building.

The automation of elevators presents a problem. Many users
(passengers) are requesting use of the same resource (elevator),
and all are seeking to minimize their wait time. A good
controller will give an overall low wait times. In this project
we use different approaches, Reinforcement Learning (RL)
and Adversarial Search (AS), to model the controller. We
evaluate the performance of each approach and compare them.

Reinforcement Learning models the controller as an agent
operating within a world. Agent types compared are Q learning,
deep Q learning and Multi Agent. Adversarial Search
models the problem as a two player game, the elevator agent
vs. the world. The agent types compared in AS are Reflex, Alpha-
Beta, and Expectimax.

The platform on which we train and test our agent is the
ElevatorSaga1 challenge. It is a coding game, which simulates
a building with an arbitrary number of floors and elevators.
The player's goal is to code the algorithm that decides the elevators
actions. Instead of writing explicit code, we train different
agents to control the elevators.

Finally, we will also compare our agents to some explicit
algorithms: random agent, Shabat elevator, and a hand crafted
solution to the ElevatorSaga game, taken from ElevatorSaga
wiki

## Problem Description
### Problem Definition
An ElevatorSaga challenge simulates a building with floors and elevators (Figure 1). Users spawn on different floors, each user has a destination floor he/ she wishes to go to. Each floor has an up button and a down button, users can press the up/ down button signaling that their destination floor is above/ below the current floor (the bottom floor and the top floor have only an up/ down button respectively). The elevators may move to any floor (including the floor they're on). Upon reaching a floor the elevator stops, and all the users in the elevator whose destination floor is the cur-rent floor will exit. Any user waiting at that floor will enter the elevator if there is room. 
![](https://github.com/ednussi/3deception/blob/master/display/figure1.PNG)
### Complexity Analysis
### Evaluation & Metrics

How To Run:

1. To run the demo on cs lab computers:
	a) install python package aiohttp ('pip3 install aiohttp --user')
	b) run 'run_aquarium.sh'

2. To run the demo on personal computer:
	a) install python 3.5
	b) install requirements (pip3 install -r requirements.txt)
	c) run 'python3 server.py'
	d) open a web browser and navigate to the link printed in console (e.g. '======== Running on http://localhost:42176 ========')

