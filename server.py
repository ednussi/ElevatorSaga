import json
import os
import webbrowser
import threading
from typing import Iterable

from aiohttp import web

from adversarialSearch import MinimaxAgent, ExpectimaxAgent, AlphaBetaAgent, ReflexAgent
from agents import *
from deepQLearning import DeepQAgent
from elevator import Elevator
from floor import Floor
from multiAgent import MultiAgent
from simulator import Simulator
from user import User


def parse_input(obj):
    obj = json.loads(list(obj.keys())[0])
    dt = None
    if 'dt' in obj:
        dt = obj['dt']
    floors = []
    for f in obj['floors']:
        users = filter(lambda u: u is not None and u['currentFloor'] == f['floorNum'], obj['world']['users'])
        users = [User(100, u['currentFloor'], u['destinationFloor'],
                      obj['world']['elapsedTime'] - u['spawnTimestamp'])
                 for u in users]
        floors.append(Floor(f['floorNum'],
                            f['buttonStates'],
                            list(set(users))))
    elevators = []
    for i, e in enumerate(obj['elevators']):
        el = Elevator(e['id'],
                      e['currentFloor'],
                      e['maxPassengerCount'],
                      e['loadFactor'],
                      e['destinationDirection'],
                      e['destinationQueue'],
                      e['pressedFloors'],
                      [{'user': None} for _ in range(e['maxPassengerCount'])],
                      len(floors))
        for u in obj['world']['elevators'][i]['userSlots']:
            if u['user'] is not None:
                user = User(100, u['user']['currentFloor'], u['user']['destinationFloor'],
                            obj['world']['elapsedTime'] - u['user']['spawnTimestamp'])
                el.load(user)
        elevators.append(el)
    userData = obj['userData']
    if 'testRun' not in userData:
        userData['testRun'] = 0
    return elevators, floors, dt, obj['world'], userData


def make_result(elevators: Iterable[Elevator], floors: Iterable[Floor], world, userData):
    result = {
        'elevators': [],
        'floors': [],
        'world': world,
        'userData': userData
    }
    for e in elevators:
        edict = {'id': e.id, 'actions': e.actions, 'events': e.eventHandles}
        result['elevators'].append(edict)
    for f in floors:
        fdict = {'id': f.floorNum, 'events': f.eventHandles}
        result['floors'].append(fdict)
    return result


async def index(_):
    return web.HTTPFound('/static/index.html#challenge=2')

def validateTrainingFinished():
    global trainThread
    if trainThread is not None:
        trainThread.join()

UPDATES_COUNT = 0

async def init(request):
    global UPDATES_COUNT
    UPDATES_COUNT = 0
    # validateTrainingFinished()
    obj = await request.post()
    elevators, floors, _, world, userData = parse_input(obj)
    agent.addMetricSample('Average Wait Time', world['avgWaitTime'], x=UPDATES_COUNT, suffix=userData['testRun'])
    state = sim.mdp.worldToState(elevators, floors, world)
    action, vals = agent.getAction(state)
    new_elevators, new_floors = sim.mdp.actionToWorld(elevators, floors, action)
    response = make_result(new_elevators, new_floors, world, userData)
    UPDATES_COUNT += 1
    return web.json_response(response)


async def update(request):
    global UPDATES_COUNT
    # validateTrainingFinished()
    obj = await request.post()
    elevators, floors, dt, world, userData = parse_input(obj)

    if sum(map(lambda e: len(e.destinationQueue), elevators)) > 0:
        # don't move elevators if they have active destination
        response = make_result(elevators, floors, world, userData)
        return web.json_response(response)

    agent.addMetricSample('Average Wait Time', world['avgWaitTime'], x=UPDATES_COUNT, suffix=userData['testRun'])
    state = sim.mdp.worldToState(elevators, floors, world)
    action, vals = agent.getAction(state)
    new_elevators, new_floors = sim.mdp.actionToWorld(elevators, floors, action)
    response = make_result(new_elevators, new_floors, world, userData)
    UPDATES_COUNT += 1
    return web.json_response(response)


async def trainMore(request):
    """Returns true if still training"""
    global trainThread
    if trainThread is not None and trainThread.isAlive():
        return web.json_response(False)

    obj = await request.post()
    global sim
    sim.iter_per_episode = int(obj['iterPerEp'])
    sim.spawn_factor = float(obj['spawnFactor'])
    sim.print('setting alpha={} eps={} gamma={} spawn={}'.format(float(obj['alpha']), float(obj['epsilon']), float(obj['gamma']), float(obj['spawnFactor'])))
    global agent
    agent.setNumTraining(int(obj['iter']))
    agent.setNumExploration(int(obj['exploreIter']))
    agent.setAlpha(float(obj['alpha']))
    agent.setEpsilon(float(obj['epsilon']))
    agent.setDiscount(float(obj['gamma']))

    trainThread = threading.Thread(target=lambda: sim.train(agent))

    trainThread.start()

    return web.json_response(False)


async def stopTraining(_):
    global sim
    global trainThread
    sim.shouldStopTraining = True
    trainThread.join()
    global agent

    return web.json_response({'status': 'STOPPED'})


async def checkTrainingStatus(_):
    global trainThread
    global agent

    if trainThread is not None and trainThread.isAlive():
        return web.json_response({
            'iterations': agent.trained if agent is not None else 0,
            'finished': False,
            'currentAlpha': agent.alpha if agent is not None else 'None',
            'currentEpsilon': agent.epsilon if agent is not None else 'None',
            'currentGamma': agent.discount if agent is not None else 'None'
        })

    return web.json_response({
        'iterations': agent.trained if agent is not None else 0,
        'finished': True,
        'currentAlpha': agent.alpha if agent is not None else 'None',
        'currentEpsilon': agent.epsilon if agent is not None else 'None',
        'currentGamma': agent.discount if agent is not None else 'None'
    })


async def plots(_):
    return web.json_response(agent.metrics if agent is not None else {})


async def changeChallenge(request):
    obj = await request.post()

    global sim
    if sim is not None:
        sim.shouldStopTraining = True

    global trainThread
    if trainThread is not None and trainThread.isAlive():
        trainThread.join()

    sim = Simulator(num_floors=int(obj['numFloors']),
                    num_elevators=int(obj['numElevators']),
                    num_iterations_per_episode=int(obj['iterPerEp']),
                    reward_user_in=float(obj['rewardInside']),
                    reward_user_out=float(obj['rewardOutside']),
                    reward_move=float(obj['rewardMovement']),
                    spawn_factor=float(obj['spawnFactor']))

    global agent
    if obj['agent'] == 'qLearning':
        agent = QLearningAgent(actionFn=sim.mdp.goToAnyFloor,
                               numTraining=60000,
                               numExploration=10000,
                               epsilon=float(obj['epsilon']),
                               alpha=float(obj['alpha']),
                               gamma=float(obj['gamma']))
    elif obj['agent'] == 'multiAgent':
        agent = MultiAgent(int(obj['numElevators']),
                           actionFn=sim.mdp.goToAnyFloorMultiAgent,
                           numTraining=60000,
                           numExploration=10000,
                           epsilon=float(obj['epsilon']),
                           alpha=float(obj['alpha']),
                           gamma=float(obj['gamma']))
    elif obj['agent'] == 'reflex':
        agent = ReflexAgent()
    elif obj['agent'] == 'minimax':
        agent = MinimaxAgent(sim, 'simpleEvaluationFunction', int(obj['searchDepth']))
    elif obj['agent'] == 'abprune':
        agent = AlphaBetaAgent(sim, 'simpleEvaluationFunction', int(obj['searchDepth']))
    elif obj['agent'] == 'expectimax':
        print('Creating ExpectimaxAgent with depth', int(obj['searchDepth']))
        agent = ExpectimaxAgent(sim, 'simpleEvaluationFunction', int(obj['searchDepth']))
    elif obj['agent'] == 'deepQAgent':
        agent = DeepQAgent(int(obj['numFloors']),
                           int(obj['numElevators']),
                           actionFn=sim.mdp.goToAnyFloor,
                           numTraining=60000,
                           numExploration=10000,
                           epsilon=float(obj['epsilon']),
                           alpha=float(obj['alpha']),
                           gamma=float(obj['gamma']))
    elif obj['agent'] == 'policyGrad':
        agent = PolicyGradientAgent(actionFn=sim.mdp.goToAnyFloor, alpha=float(obj['alpha']), epsilon=float(obj['epsilon']), gamma=float(obj['gamma']), numTraining=50000, numExploration=40000, floors=obj['numFloors'], iterPerEp=int(obj['iterPerEp']))
    elif obj['agent'] == 'random':
        agent = RandomAgent(sim.num_floors, sim.num_elevators)
    elif obj['agent'] == 'shabbat':
        agent = ShabatAgent(sim.num_floors, sim.num_elevators)

    global loadedModels
    if agent.__class__.__name__ in loadedModels:
        agent.loadModel(loadedModels[agent.__class__.__name__])

    return web.json_response({
        'iterations': agent.trained,
        'alpha': agent.alpha,
        'epsilon': agent.epsilon,
        'gamma': agent.discount,
        'iterPerEp': sim.iter_per_episode,
        'rewardInside': sim.mdp.reward_user_in,
        'rewardOutside': sim.mdp.reward_user_out,
        'rewardMovement': sim.mdp.reward_move,
        'spawnFactor': sim.spawn_factor,
    })


async def logs(_):
    if sim is not None and agent is not None:
        simlogs = sim.logs[:]
        sim.logs = []
        aglogs = agent.logs[:]
        agent.logs = []
        return web.json_response({'sim': simlogs, 'agent': aglogs})
    return web.json_response({'sim': [], 'agent': []})


async def getSavedModels(_):
    global agent
    global sim
    if not os.path.isdir(os.path.join(os.getcwd(), 'models', agent.__class__.__name__)):
        os.mkdir(os.path.join(os.getcwd(), 'models', agent.__class__.__name__))
        return web.json_response([])

    exps = os.listdir(os.path.join(os.getcwd(), 'models', agent.__class__.__name__))
    res = []
    for exp in exps:
        conf = await parse_hyperparams(exp)
        if int(conf['E']) == sim.num_elevators and int(conf['F']) == sim.num_floors and conf['N'] == agent.__class__.__name__:
            res.append(conf)

    return web.json_response(res)


async def parse_hyperparams(exp):
    conf = {'_id': exp}
    confstr = exp[:-4]
    confp = confstr.split('_')
    for x in confp:
        k, v = x.split('=')
        conf[k] = v
    return conf


async def loadModel(request):
    obj = await request.post()
    global agent
    # sim.print('loading model: {}'.format(obj['modelName']))
    modelPath = os.path.join(os.getcwd(), 'models', agent.__class__.__name__, obj['_id'])
    agent.loadModel(modelPath)

    global loadedModels
    loadedModels[agent.__class__.__name__] = modelPath
    return web.json_response('ok')


async def loadBestModel(request):
    obj = await request.post()
    agentName = obj['agent']
    sim.print('loading best model for agent: {}'.format(obj['agent']))

    if agentName in ['qLearning', 'multiAgent', 'deepQAgent', 'policyGrad']:
        if not os.path.isdir(os.path.join(os.getcwd(), 'best_models')):
            os.mkdir(os.path.join(os.getcwd(), 'best_models'))
        if not os.path.isdir(os.path.join(os.getcwd(), 'best_models', agentName)):
            os.mkdir(os.path.join(os.getcwd(), 'best_models', agentName))

        modelPath = os.listdir(os.path.join(os.getcwd(), 'best_models', obj['agent']))
        if not len(modelPath):
            sim.print('best model does not exist!')
            return web.json_response('Best model does not exist!')

        realModelPath = None
        for mp in modelPath:
            conf = await parse_hyperparams(mp)
            if conf['E'] == obj['numElevators'] and conf['F'] == obj['numFloors']:
                realModelPath = mp
                break
        if realModelPath is None:
            sim.print('best model does not exist!')
            return web.json_response('Best model does not exist!')

        conf = await parse_hyperparams(realModelPath)

        global agent
        if agentName == 'qLearning':
            agent = QLearningAgent(actionFn=sim.mdp.goToAnyFloor)
        elif agentName == 'multiAgent':
            agent = MultiAgent(int(conf['E']), actionFn=sim.mdp.goToAnyFloorMultiAgent)
        elif agentName == 'deepQAgent':
            agent = DeepQAgent(int(conf['F']), int(conf['E']), actionFn=sim.mdp.goToAnyFloor)

        fullRealModelPath = os.path.join(os.getcwd(), 'best_models', obj['agent'], realModelPath)
        agent.loadModel(fullRealModelPath)
        global loadedModels
        loadedModels[agent.__class__.__name__] = fullRealModelPath

    else:
        if agentName == 'reflex':
            agent = ReflexAgent()
        elif agentName == 'abprune2':
            agent = AlphaBetaAgent(sim, 'simpleEvaluationFunction', 2)
        elif agentName == 'abprune3':
            agent = AlphaBetaAgent(sim, 'simpleEvaluationFunction', 3)
        elif agentName == 'expectimax2':
            agent = ExpectimaxAgent(sim, 'simpleEvaluationFunction', 2)
        elif agentName == 'expectimax3':
            agent = ExpectimaxAgent(sim, 'simpleEvaluationFunction', 3)
        elif agentName == 'random':
            agent = RandomAgent(sim.num_floors, sim.num_elevators)
        elif agentName == 'shabbat':
            agent = ShabatAgent(sim.num_floors, sim.num_elevators)

    return web.json_response('Ready!')


if __name__ == '__main__':
    sim = None
    agent = None
    trainThread = None
    loadedModels = {}

    serv = web.Application()
    serv.add_routes([
        web.get('/', index),
        web.get('/checkTrainingStatus', checkTrainingStatus),
        web.get('/stopTraining', stopTraining),
        web.get('/logs', logs),
        web.get('/exps', getSavedModels),
        web.post('/plots', plots),
        web.post('/train', trainMore),
        web.post('/init', init),
        web.post('/update', update),
        web.post('/changeChallenge', changeChallenge),
        web.post('/loadModel', loadModel),
        web.post('/loadBestModel', loadBestModel),
        web.static('/static', './static')
    ])
    host = 'localhost'
    port = random.randint(10000, 65500)
    os.system('chromium http://localhost:' + str(port))
    web.run_app(serv, host=host, port=port)
