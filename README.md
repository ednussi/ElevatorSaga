# ElevatorSaga

## Reinforcement learning control agent for Elevators, focusing on minimizing average wait time and maximizing throuput in a large stochastic enviorment

By: Eran Nussinovitch, Gregory Pasternak, Asaph Shamir

## I. Introduction
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

## II. Problem Description
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
We use a realistic approach, in which our agent only sees information available for a controller in real life. That is, it knows which buttons are pressed in each floor and in the elevators, and it knows the elevator locations. The agent does not know how many users are waiting on each floor and how many users are inside each elevator. However, the agent does know if an elevator is full, simulating an overload sensor which lets the elevator know when it is too heavy to move. Such sensors exist in elevators today. 

For each floor, excluding the top, there is an up button which can be pressed or not pressed, giving <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-1}" title="2^{N-1}" /></a> combinations. 

Each floor besides the bottom has a down button which also gives <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-1}" title="2^{N-1}" /></a> combination, in total, <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{2N-2}" title="2^{2N-2}" /></a> cases.
Within every elevator, each floor button could be pressed or not, beside the floor the elevator is currently in, giving 2 (𝑁−1) cases for each elevator. In total: <a href="https://www.codecogs.com/eqnedit.php?latex=(2^{N-1})^{M}=2^{M(N-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(2^{N-1})^{M}=2^{M(N-1)}" title="(2^{N-1})^{M}=2^{M(N-1)}" /></a>. 

Each elevator can be in any given floor so we have <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N^{M}" title="N^{M}" /></a> combinations for locations. In addition, the load factor of each elevator is represented as a single Boolean giving in total <a href="https://www.codecogs.com/eqnedit.php?latex=2^{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{M}" title="2^{M}" /></a> cases.

All of the above gives us a state space of size:

<a href="https://www.codecogs.com/eqnedit.php?latex=2^{(2&plus;M)\cdot&space;(N-1)}&space;\cdot&space;N^{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{(2&plus;M)\cdot&space;(N-1)}&space;\cdot&space;N^{M}" title="2^{(2+M)\cdot (N-1)} \cdot N^{M}" /></a>

#### Action space
Each elevator can go to any floor, including to stay on the floor it is on right now, resulting in <a href="https://www.codecogs.com/eqnedit.php?latex=2^{N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N^{M}" title="N^{M}" /></a> possible actions

#### State-Action space
Considering all the possible states and actions, the full space size is:

<a href="https://www.codecogs.com/eqnedit.php?latex=2^{(2&plus;M)\cdot&space;(N-1)}&space;\cdot&space;N^{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{(2&plus;M)\cdot&space;(N-1)}&space;\cdot&space;N^{2M}" title="2^{(2+M)\cdot (N-1)} \cdot N^{2M}" /></a>

The size of the state-action space is exponential in both the number of floors and the number of elevators, and is the main obstacle for our agents. For example, Rothberg building B front side has 2 elevators, each of them can reach 7 floors (- 2, -1 and 1 through 5) giving a state-action space of size ≈ <a href="https://www.codecogs.com/eqnedit.php?latex=2^{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{35}" title="2^{35}" /></a>

### Evaluation & Metrics
We define the wait time of a user as the time passed since the user spawned until the user exits an elevator at the destination floor. Two metrics are used in our evaluation:
* Average user wait time
* Maximal user wait time 

Average wait time reflects overall model performance, while maximal wait time shows fairness of the model. If a model shows significantly higher maximal wait times, it means there were users that experience starvation, which should be avoided.

### Scenrios 
We tested our agents on three scenarios (buildings):
* 3 floors, 1 elevator
* 7 floors, 1 elevator
* 7 floors, 2 elevators (Rothberg building B)

## III. Solution Approaches
### Reinforcement Learning
Due to the Markovian nature of the world (past states are insignificant when deciding on an action), Value Estimation is the most natural go-to approach. However, our world is stochastic, and the agent doesn't know the probability function for state-action-state (we don’t know when and where users will be generated). This makes RL, and particularly Q-learning, a model-free agent, most suitable for this problem. Even though Q-learning is model free, it still suffers from the huge state-action space in this problem. To lessen this we unified the up and down buttons in each floor to a single "call elevator" button reducing the number of state by a factor of <a href="https://www.codecogs.com/eqnedit.php?latex=2^{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N-2}" title="2^{N-2}" /></a>

The space size is particularly problematic for the scenario with 7 floors and 2 elevators. The state-action space is so large that it is computationally impossible to use regular Qlearning. We therefore use two variations of Q-learning for this scenario: Deep-Q-learning using a neural network and Multi Agent Q learning (see IV for further details). 

Lastly, an RL algorithm requires feedback from the environment. Since the ElevatorSaga game does not have rewards, we design a reward system that supports the training process of our algorithms. 

The reward for a state-action-nextState triplet is made up of several parts:
Let <a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}^{exit}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{i}^{exit}" title="n_{i}^{exit}" /></a> be the number of users exiting elevator 𝑖 after it moved, then the reward for exiting users is:

<a href="https://www.codecogs.com/eqnedit.php?latex=r_{exit}=\sum_{i=1}^{M}n_{i}^{exit}\cdot&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{exit}=\sum_{i=1}^{M}n_{i}^{exit}\cdot&space;2" title="r_{exit}=\sum_{i=1}^{M}n_{i}^{exit}\cdot 2" /></a>

The movement penalty for each elevator is given by:

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}^{move}=\left&space;(&space;\left&space;(&space;\left&space;(&space;a_i-l_i&space;\right&space;)&space;-1&space;\right&space;)&space;\cdot&space;0.3&space;&plus;&space;0.4&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}^{move}=\left&space;(&space;\left&space;(&space;\left&space;(&space;a_i-l_i&space;\right&space;)&space;-1&space;\right&space;)&space;\cdot&space;0.3&space;&plus;&space;0.4&space;\right&space;)" title="p_{i}^{move}=\left ( \left ( \left ( a_i-l_i \right ) -1 \right ) \cdot 0.3 + 0.4 \right )" /></a>

Where <a href="https://www.codecogs.com/eqnedit.php?latex=a_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_i" title="a_i" /></a> is the floor elevator 𝑖 reached after moving, and <a href="https://www.codecogs.com/eqnedit.php?latex=a_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_i" title="l_i" /></a> is the floor it left. 

The total movement penalty is:

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{move}=\sum_{i=1}^{M}&space;p_{i}^{move}&space;\cdot&space;-\frac{1}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{move}=\sum_{i=1}^{M}&space;p_{i}^{move}&space;\cdot&space;-\frac{1}{2}" title="p_{move}=\sum_{i=1}^{M} p_{i}^{move} \cdot -\frac{1}{2}" /></a>

The users' penalty is given by:

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{users}=-1\cdot\left&space;(&space;\sum_{i=1}^{M}&space;n_{i}^{stay}&space;&plus;&space;min\left&space;(&space;M\cdot&space;C,&space;\sum_{j=1}^{N}&space;n_{j}^{wait}&space;\right&space;)\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{users}=-1\cdot\left&space;(&space;\sum_{i=1}^{M}&space;n_{i}^{stay}&space;&plus;&space;min\left&space;(&space;M\cdot&space;C,&space;\sum_{j=1}^{N}&space;n_{j}^{wait}&space;\right&space;)\right&space;)" title="p_{users}=-1\cdot\left ( \sum_{i=1}^{M} n_{i}^{stay} + min\left ( M\cdot C, \sum_{j=1}^{N} n_{j}^{wait} \right )\right )" /></a>

Where <a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}^{wait}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{i}^{stay}" title="n_{i}^{stay}" /></a> is the number of users still in elevator 𝑖 after it moved, 𝐶 is the capacity of the elevators and <a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}^{wait}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{j}^{wait}" title="n_{j}^{wait}" /></a>  is the number of users waiting in floor j.

Additionally, we have special penalties. Let 𝐹 be the set of floors with "call" buttons pressed. Let <a href="https://www.codecogs.com/eqnedit.php?latex=b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_i" title="b_i" /></a> be the set of buttons pressed in elevator 𝑖 (the floors requested by users in that elevator). Let <a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}^{in}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{i}^{in}" title="n_{i}^{in}" /></a> be the number of users in elevator 𝑖 before it moved. Then we have:

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}^{full}=\left\{\begin{matrix}&space;-200&space;&&space;,n_{i}^{in}=C\wedge&space;a_i\notin&space;b_i&space;\\&space;0&space;&&space;,else&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}^{full}=\left\{\begin{matrix}&space;-200&space;&&space;,n_{i}^{in}=C\wedge&space;a_i\notin&space;b_i&space;\\&space;0&space;&&space;,else&space;\end{matrix}\right." title="p_{i}^{full}=\left\{\begin{matrix} -200 & ,n_{i}^{in}=C\wedge a_i\notin b_i \\ 0 & ,else \end{matrix}\right." /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}^{empty}=\left\{\begin{matrix}&space;-200&space;&&space;,n_{i}^{in}=0\wedge&space;a_i\notin&space;F&space;\wedge&space;F\neq&space;0&space;\\&space;0&space;&&space;,else&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}^{empty}=\left\{\begin{matrix}&space;-200&space;&&space;,n_{i}^{in}=0\wedge&space;a_i\notin&space;F&space;\wedge&space;F\neq&space;0&space;\\&space;0&space;&&space;,else&space;\end{matrix}\right." title="p_{i}^{empty}=\left\{\begin{matrix} -200 & ,n_{i}^{in}=0\wedge a_i\notin F \wedge F\neq 0 \\ 0 & ,else \end{matrix}\right." /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}^{bad}=\left\{\begin{matrix}&space;-100&space;&&space;,0<n_{i}^{in}<C\wedge&space;a_i\notin&space;F&space;\cup&space;b_i&space;\\&space;0&space;&&space;,else&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}^{bad}=\left\{\begin{matrix}&space;-100&space;&&space;,0<n_{i}^{in}<C\wedge&space;a_i\notin&space;F&space;\cup&space;b_i&space;\\&space;0&space;&&space;,else&space;\end{matrix}\right." title="p_{i}^{bad}=\left\{\begin{matrix} -100 & ,0<n_{i}^{in}<C\wedge a_i\notin F \cup b_i \\ 0 & ,else \end{matrix}\right." /></a>

The total reward is given by:

<a href="https://www.codecogs.com/eqnedit.php?latex=R&space;=&space;r_{exit}&space;&plus;&space;p_{move}&space;&plus;&space;p_{users}&space;&plus;&space;\sum_{i=1}^{M}&space;p_{i}^{full}&space;&plus;&space;p_{i}^{empty}&space;&plus;&space;p_{i}^{bad}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R&space;=&space;r_{exit}&space;&plus;&space;p_{move}&space;&plus;&space;p_{users}&space;&plus;&space;\sum_{i=1}^{M}&space;p_{i}^{full}&space;&plus;&space;p_{i}^{empty}&space;&plus;&space;p_{i}^{bad}" title="R = r_{exit} + p_{move} + p_{users} + \sum_{i=1}^{M} p_{i}^{full} + p_{i}^{empty} + p_{i}^{bad}" /></a>

The design of the reward system plays a major part in determining the performance of our agents, and is further discussed in VI.1.

### Adversarial Search
We note that this problem could be modeled as a 2-player game where the first (max) player is the elevator agent and the second player is the world. The world player always chooses an action by some stochastic process. Like Q-Learning, adversarial search agents suffer greatly from the state-action space size which translates directly to a very large branching factor. This limits greatly the depth to which an AS agent can go during runtime. The elevator should operate in real-time, i.e. the agent's computation time to decide on the next action should seem instantaneous to the user. 

For AS agents, the utility of a state is computed using simple evaluation function (adversarialSearch.py:340):

<a href="https://www.codecogs.com/eqnedit.php?latex=u=\&hash;users\&space;outside\&space;the\&space;elevator&space;\cdot&space;(-1.1)&space;&plus;&space;\&hash;users\&space;inside\&space;the\&space;elevator&space;\cdot&space;(-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u=\&hash;users\&space;outside\&space;the\&space;elevator&space;\cdot&space;(-1.1)&space;&plus;&space;\&hash;users\&space;inside\&space;the\&space;elevator&space;\cdot&space;(-1)" title="u=\#users\ outside\ the\ elevator \cdot (-1.1) + \#users\ inside\ the\ elevator \cdot (-1)" /></a>

Note: when a passenger arrives to its destination, he is immediately removed from the world, and stops being counted in consecutive rewards

## Implementation Details
### Reinforcement Learning Agents
1. A numbered list
aba
2. Which is numbered
aba
3. test
aba

How To Run:

1. To run the demo on cs lab computers:
	a) install python package aiohttp ('pip3 install aiohttp --user')
	b) run 'run_aquarium.sh'

2. To run the demo on personal computer:
	a) install python 3.5
	b) install requirements (pip3 install -r requirements.txt)
	c) run 'python3 server.py'
	d) open a web browser and navigate to the link printed in console (e.g. '======== Running on http://localhost:42176 ========')

