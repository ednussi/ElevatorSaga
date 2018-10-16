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

Figure 1: Example of an ElevatorSaga challenge with 8 floors and 2 elevators. The down button on the 5th floor is pressed. The 7th floor button in the second elevator is pressed.

We begin the formalization of the elevator saga game by defining:

<a href="https://www.codecogs.com/eqnedit.php?latex=|FLOORS|=N,&space;|ELEVATORS|=M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|FLOORS|=N,&space;|ELEVATORS|=M" title="|FLOORS|=N, |ELEVATORS|=M" /></a>

In each run of the simulator there is a spawn factor 𝑝𝑠 governing the generation of new users. In every time step, a user is generated with probability 𝑝𝑠. The probability for creating a user in a certain floor is: 

<a href="https://www.codecogs.com/eqnedit.php?latex=p(floor)=\left\{\begin{matrix}&space;\frac{1}{2}&plus;\frac{\frac{1}{2}}{N}&space;&&space;,floor=0\\&space;\frac{\frac{1}{2}}{N}&space;&&space;,floor>0&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(floor)=\left\{\begin{matrix}&space;\frac{1}{2}&plus;\frac{\frac{1}{2}}{N}&space;&&space;,floor=0\\&space;\frac{\frac{1}{2}}{N}&space;&&space;,floor>0&space;\end{matrix}\right." title="p(floor)=\left\{\begin{matrix} \frac{1}{2}+\frac{\frac{1}{2}}{N} & ,floor=0\\ \frac{\frac{1}{2}}{N} & ,floor>0 \end{matrix}\right." /></a>

Meaning most users spawn at floor 0. If a user spawns at floor 0 its destination floor is randomly sampled out of {1, … , 𝑁 −1}. Otherwise 90% of the time the user destination floor is 0, and 10% of the time it is randomly selected out of the other floors. Each elevator has a maximum capacity. In our scenarios all elevators capacities were set to 4. An elevator with 4 users inside is full, and floor users cannot enter the elevator until users inside the elevator exit.

### Complexity Analysis
#### State space
We use a realistic approach, in which our agent only sees information available for a controller in real life. That is, it knows which buttons are pressed in each floor and in the elevators, and it knows the elevator locations. The agent does not know how many users are waiting on each floor and how many users are inside each elevator. However, the agent does know if an elevator is full, simulating an overload sensor which lets the elevator know when it is too heavy to move. Such sensors exist in elevators today. For each floor, excluding the top, there is an up button which can be pressed or not pressed, giving <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-1}" title="2^{N-1}" /></a> combinations. Each floor besides the bottom has a down button which also gives <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-1}" title="2^{N-1}" /></a> combination, in total, <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-1}" title="2^{2N-2}" /></a> cases.
Within every elevator, each floor button could be pressed or not, beside the floor the elevator is currently in, giving 2 (𝑁−1) cases for each elevator. In total: <a href="https://www.codecogs.com/eqnedit.php?latex=(2^{N-1})^{M}=2^{M(N-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(2^{N-1})^{M}=2^{M(N-1)}" title="(2^{N-1})^{M}=2^{M(N-1)}" /></a>. Each elevator can be in any given floor so we have <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-1}" title="N^{M}" /></a> combinations for locations. In addition, the load factor of each elevator is represented as a single Boolean giving in total <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{M}" title="2^{M}" /></a> cases.

All of the above gives us a state space of size:

<a href="https://www.codecogs.com/eqnedit.php?latex=2^{(2&plus;M)\cdot&space;(N-1)}&space;\cdot&space;N^{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{(2&plus;M)\cdot&space;(N-1)}&space;\cdot&space;N^{M}" title="2^{(2+M)\cdot (N-1)} \cdot N^{M}" /></a>

#### Action space
#### State-Action space

### Evaluation & Metrics
## Solution Approaches

How To Run:

1. To run the demo on cs lab computers:
	a) install python package aiohttp ('pip3 install aiohttp --user')
	b) run 'run_aquarium.sh'

2. To run the demo on personal computer:
	a) install python 3.5
	b) install requirements (pip3 install -r requirements.txt)
	c) run 'python3 server.py'
	d) open a web browser and navigate to the link printed in console (e.g. '======== Running on http://localhost:42176 ========')

