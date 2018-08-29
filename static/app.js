var UPDATE_EVERY = 1;

var createParamsUrl = function(current, overrides) {
    return "#" + _.map(_.merge(current, overrides), function(val, key) {
        return key + "=" + val;
    }).join(",");
};

var cache = [];
var serializer = function(key, value) {
    if (typeof value === 'object' && value !== null) {
        if (cache.indexOf(value) !== -1) {
            // Duplicate reference found
            try {
                // If this value does not reference a parent it can be deduped
                return JSON.parse(JSON.stringify(value));
            } catch (error) {
                // discard key if value cannot be deduped
                return;
            }
        }
        // Store value in our collection
        cache.push(value);
    }
    return value;
};

$(function() {
    var tsKey = "elevatorTimeScale";

    var params = {};

    var trainMoreButton = $("#trainMore");
    trainMoreButton.click(function() {
        var trainIter = $('#trainIter').val();
        var exploreIter = $('#exploreIter').val();
        var iterPerEp = $('#iterPerEp').val();
        //        app.logsTimer = setInterval(app.logsHandler, 500);
        trainMoreButton.hide();
        stopTrainingButton.show();
        trainMoreButton.text('Training...');
        trainMoreButton.attr('disabled', 'disabled');
        $('#alpha').attr('disabled', 'disabled');
        $('#epsilon').attr('disabled', 'disabled');
        $('#gamma').attr('disabled', 'disabled');
        $('#trainIter').attr('disabled', 'disabled');
        $('#exploreIter').attr('disabled', 'disabled');
        $('#iterPerEp').attr('disabled', 'disabled');
        $('#rewardInside').attr('disabled', 'disabled');
        $('#rewardOutside').attr('disabled', 'disabled');
        $('#rewardMovement').attr('disabled', 'disabled');
        $('#spawnFactor').attr('disabled', 'disabled');
        $.ajax({
            method: 'POST',
            url: '/train',
            data: {
                iter: trainIter,
                exploreIter: exploreIter,
                iterPerEp: iterPerEp,
                alpha: $('#alpha').val(),
                epsilon: $('#epsilon').val(),
                gamma: $('#gamma').val(),
                spawnFactor: $('#spawnFactor').val(),
            },
            success: function(res) {}
        });
        app.trainTimer = setInterval(function() {
            $.get('/checkTrainingStatus', function(res) {
                $('#trained').text(res['iterations']);
                if (!res['finished']) {
                    $('#alpha').val(res['currentAlpha']);
                    $('#epsilon').val(res['currentEpsilon']);
                    $('#gamma').val(res['currentGamma']);
                } else {
                    trainMoreButton.show();
                    stopTrainingButton.hide();
                    trainMoreButton.text('Train');
                    trainMoreButton.removeAttr('disabled');
                    $('#alpha').removeAttr('disabled');
                    $('#epsilon').removeAttr('disabled');
                    $('#gamma').removeAttr('disabled');
                    $('#trainIter').removeAttr('disabled');
                    $('#exploreIter').removeAttr('disabled');
                    $('#iterPerEp').removeAttr('disabled');
                    $('#rewardInside').removeAttr('disabled');
                    $('#rewardOutside').removeAttr('disabled');
                    $('#rewardMovement').removeAttr('disabled');
                    $('#spawnFactor').removeAttr('disabled');
                    clearInterval(app.trainTimer);
                    //                    clearInterval(app.logsTimer);
                }
            })
        }, 1000);
    });

    var stopTrainingButton = $("#stopTraining");
    stopTrainingButton.hide();
    stopTrainingButton.click(function() {
        if (app.trainTimer) {
            clearInterval(app.trainTimer);
        }
        $.ajax({
            method: 'GET',
            url: '/stopTraining',
            success: function(res) {
                trainMoreButton.show();
                stopTrainingButton.hide();
                trainMoreButton.text('Train');
                trainMoreButton.removeAttr('disabled');
                $('#alpha').removeAttr('disabled');
                $('#epsilon').removeAttr('disabled');
                $('#gamma').removeAttr('disabled');
                $('#trainIter').removeAttr('disabled');
                $('#exploreIter').removeAttr('disabled');
                $('#iterPerEp').removeAttr('disabled');
                $('#rewardInside').removeAttr('disabled');
                $('#rewardOutside').removeAttr('disabled');
                $('#rewardMovement').removeAttr('disabled');
            }
        });
    });

    var $world = $(".innerworld");
    var $stats = $(".statscontainer");
    var $feedback = $(".feedbackcontainer");
    var $challenge = $(".challenge");
    var $codestatus = $(".codestatus");

    var floorTempl = document.getElementById("floor-template").innerHTML.trim();
    var elevatorTempl = document.getElementById("elevator-template").innerHTML.trim();
    var elevatorButtonTempl = document.getElementById("elevatorbutton-template").innerHTML.trim();
    var userTempl = document.getElementById("user-template").innerHTML.trim();
    var challengeTempl = document.getElementById("challenge-template").innerHTML.trim();
    var feedbackTempl = document.getElementById("feedback-template").innerHTML.trim();
    var codeStatusTempl = document.getElementById("codestatus-template").innerHTML.trim();

    var app = riot.observable({});

    app.loadBestModel = function(agent) {
        $("#bestAgentLoaded").text('Loading...');
        $("#bestAgentLoaded").fadeIn();
        var ch = app.predefChallenges[parseInt($("#chooseChallenge option:selected").val(), 10)];
        $.ajax({
            method: 'POST',
            url: '/loadBestModel',
            data: {
                agent: agent,
                numFloors: ch[0],
                numElevators: ch[1]
            },
            success: function(res) {
                $("#bestAgentLoaded").text(res);
                setTimeout(function() {
                    $("#bestAgentLoaded").fadeOut();
                }, 1000);
            },
            dataType: 'text',
            async: true
        });
    };

    app.showViableAgents = function(ch) {
        var ag = null;

        switch (ch) {
            case 1:
                ag = {
                    random: "Random",
                    shabbat: "Shabat",
                    reflex: "Reflex",
                    abprune2: "A-B Pruning (depth 2)",
                    abprune3: "A-B Pruning (depth 3)",
                    expectimax2: "Expectimax (depth 2)",
                    expectimax3: "Expectimax (depth 3)",
                    qLearning: "Q Learning",
                    deepQAgent: "Deep Q Agent",
                };
                break;
            case 2:
                ag = {
                    random: "Random",
                    shabbat: "Shabat",
                    reflex: "Reflex",
                    abprune2: "A-B Pruning (depth 2)",
                    abprune3: "A-B Pruning (depth 3)",
                    qLearning: "Q Learning",
                    deepQAgent: "Deep Q Agent"
                };
                break;
            case 3:
                ag = {
                    random: "Random",
                    shabbat: "Shabat",
                    reflex: "Reflex",
                    multiAgent: "Multi-Agent Q Learning",
                    deepQAgent: "Deep Q Agent"
                };
                break;
        }

        $('#chooseBestAgent option').remove();

        var optObj = $('<option></option>');
        optObj.attr('value', 'NULL');
        optObj.text("Choose pretrained agent...");
        $('#chooseBestAgent').append(optObj);

        for (var opt of Object.keys(ag)) {
            var optObj = $('<option></option>');
            optObj.attr('value', opt);
            optObj.text(ag[opt]);
            $('#chooseBestAgent').append(optObj);
        }
    }

    app.predefChallenges = {
        1: [3, 1],
        2: [7, 1],
        3: [7, 2]
    };

    $("#chooseChallenge").change(function() {
        var ch = parseInt($("#chooseChallenge option:selected").val(), 10);
        app.showViableAgents(ch);
        app.changeChallenge(ch, ...app.predefChallenges[ch]);
    });

    $("#chooseBestAgent").change(function() {
        var agent = $("#chooseBestAgent option:selected").val();
        if (agent === 'NULL') {
            return;
        }
        var ch = parseInt($("#chooseChallenge option:selected").val(), 10);
        app.changeChallenge(ch, ...app.predefChallenges[ch]);
        app.loadBestModel(agent);
    });

    app.trainTimer = null;
    app.updatesCount = 0;
    app.worldController = createWorldController(1.0 / 60.0);
    app.worldController.on("usercode_error", function(e) {
        console.log("World raised code error", e);
    });

    app.userData = {};

    app.createMessage = function(elevators, floors) {
        return {
            elevators: elevators.map((e, i) => {
                return {
                    id: i,
                    currentFloor: e.currentFloor(),
                    goingUpIndicator: e.goingUpIndicator(),
                    goingDownIndicator: e.goingDownIndicator(),
                    maxPassengerCount: e.maxPassengerCount(),
                    loadFactor: e.loadFactor(),
                    destinationDirection: e.destinationDirection(),
                    destinationQueue: e.destinationQueue,
                    pressedFloors: e.getPressedFloors(),
                    numPassengers: app.world.elevators[i].userSlots.filter(s => s.user != null).length
                }
            }),
            floors: floors.map((f, i) => {
                return {
                    id: i,
                    floorNum: f.level,
                    buttonStates: f.buttonStates
                };
            }),
            world: app.world,
            userData: app.userData
        }
    };

    app.updatePolicy = (elevators, floors) => (response) => {
        var obj = JSON.parse(response);
        _.each(obj.elevators, (eobj) => {
            var elevator = elevators[eobj.id];
            _.each(Object.keys(eobj.events), (ev) => {
                var evhander = 'function elevatorHandler' + eobj.id + '() {';
                _.each(eobj.events[ev], (h) => {
                    evhander += 'elevators[eid].' + h + ';';
                });
                evhander += '}; return elevatorHandler' + eobj.id + ';';
                elevator.on(ev, new Function('elevators', 'floors', 'eid', evhander)(elevators, floors, eobj.id));
            });
            _.each(eobj.actions, (a) => {
                new Function('elevators', 'floors', 'eid', 'elevators[eid].' + a + ';')(elevators, floors, eobj.id);
            });
        });
        _.each(obj.floors, (fobj) => {
            var floor = floors[fobj.id];
            _.each(Object.keys(fobj.events), (ev) => {
                var evhander = '';
                _.each(fobj.events[ev], (h) => {
                    evhander += 'floors[fid].' + h + ';';
                });
                floor.on(ev, new Function('elevators', 'floors', 'fid', evhander))(elevators, floors, fobj.id);
            });
        });
    };

    app.initFunc = function(elevators, floors) {
        cache = [];
        var message = app.createMessage(elevators, floors);

        $.ajax({
            method: 'POST',
            url: '/init',
            data: JSON.stringify(message, serializer),
            success: app.updatePolicy(elevators, floors),
            dataType: 'text',
            async: false
        });
    };

    app.updateFunc = function(dt, elevators, floors) {
        if (elevators.map(e => e.destinationQueue.length).reduce((a, b) => a + b) > 0) {
            return;
        }

        cache = [];
        app.updatesCount++;
        if (app.updatesCount % UPDATE_EVERY === 0) {

            var message = app.createMessage(elevators, floors);
            message.dt = dt;

            $.ajax({
                method: 'POST',
                url: '/update',
                data: JSON.stringify(message, serializer),
                success: app.updatePolicy(elevators, floors),
                dataType: 'text',
                async: false
            });

        }
    };

    app.worldCreator = createWorldCreator();
    app.world = undefined;

    app.currentChallengeIndex = 0;

    app.startStopOrRestart = function() {
        if (app.world.challengeEnded) {
            app.startChallenge(app.currentChallengeIndex);
        } else {
            app.worldController.setPaused(!app.worldController.isPaused);
        }
    };

    app.testResults = [];
    app.testCount = 0;

    app.startChallenge = function(challengeIndex, autoStart, testMode) {
        if (testMode) {
            app.userData.testRun = app.testCount;
            $("#test").attr('disabled', 'disabled');
        }

        if (typeof app.world !== "undefined") {
            app.world.unWind();
            // TODO: Investigate if memory leaks happen here
        }
        app.currentChallengeIndex = challengeIndex;
        app.world = app.worldCreator.createWorld(challenges[challengeIndex].options);
        window.world = app.world;

        clearAll([$world, $feedback]);
        presentStats($stats, app.world);
        presentChallenge($challenge, challenges[challengeIndex], app, app.world, app.worldController, challengeIndex + 1, challengeTempl);
        presentWorld($world, app.world, floorTempl, elevatorTempl, elevatorButtonTempl, userTempl);

        app.worldController.on("timescale_changed", function() {
            localStorage.setItem(tsKey, app.worldController.timeScale);
            if (!testMode) presentChallenge($challenge, challenges[challengeIndex], app, app.world, app.worldController, challengeIndex + 1, challengeTempl);
        });

        app.world.on("stats_changed", function() {
            var challengeStatus = challenges[challengeIndex].condition.evaluate(app.world);
            if (challengeStatus !== null) {
                app.world.challengeEnded = true;
                app.worldController.setPaused(true);
                if (!testMode) {
                    if (challengeStatus) {
                        presentFeedback($feedback, feedbackTempl, app.world, "Success!", "Challenge completed", createParamsUrl(params, { challenge: (challengeIndex + 2) }));
                    } else {
                        presentFeedback($feedback, feedbackTempl, app.world, "Challenge failed", "Maybe your program needs an improvement?", "");
                    }
                } else {
                    app.testResults.push(app.world.avgWaitTime);
                    if (++app.testCount < 20) {
                        app.userData.testRun = app.testCount;
                        $("#test").text('Test (' + (20 - app.testCount) + ' runs left)');
                        setTimeout(function() {
                            app.startChallenge(challengeIndex, true, true);
                        });
                    } else {
                        var avg = app.testResults.reduce(function(a, b) { return a + b; }) / app.testResults.length;
                        presentFeedback($feedback, feedbackTempl, app.world, "Testing finished! Average: " + avg,
                            "Average wait times: " + JSON.stringify(app.testResults.map(function(e) { return e.toFixed(2); })), "");
                        app.testResults = [];
                        app.testCount = 0;
                        $("#test").text('Test (20 runs)');
                        $("#test").removeAttr('disabled');
                    }
                }
            }
        });

        var codeObj = {
            init: app.initFunc,
            update: app.updateFunc
        };
        console.log("Starting...");
        app.worldController.start(app.world, codeObj, window.requestAnimationFrame, autoStart);
    };

    riot.route(function(path) {
        params = _.reduce(path.split(","), function(result, p) {
            var match = p.match(/(\w+)=(\w+$)/);
            if (match) { result[match[1]] = match[2]; }
            return result;
        }, {});
        var requestedChallenge = 0;
        var autoStart = false;
        var timeScale = parseFloat(localStorage.getItem(tsKey)) || 2.0;
        _.each(params, function(val, key) {
            if (key === "challenge") {
                requestedChallenge = _.parseInt(val) - 1;
                if (requestedChallenge < 0 || requestedChallenge >= challenges.length) {
                    console.log("Invalid challenge index", requestedChallenge);
                    console.log("Defaulting to first challenge");
                    requestedChallenge = 0;
                }
            } else if (key === "autostart") {
                autoStart = val === "false" ? false : true;
            } else if (key === "timescale") {
                timeScale = parseFloat(val);
            } else if (key === "fullscreen") {
                makeDemoFullscreen();
            }
        });
        app.worldController.setTimeScale(timeScale);
        app.startChallenge(requestedChallenge, autoStart);
    });

    app.ks = {};

    app.drawPlots = function(res) {
        var layout = {
            autosize: false,
            width: 380,
            height: 380,
            margin: {
                l: 50,
                r: 20,
                b: 30,
                t: 50,
                pad: 4
            }
        };

        Object.keys(res).sort().forEach((k, i) => {
            var parts = k.split(':');

            var el = $('[plotName=' + parts[0] + ']')[0];
            if (!el) {
                el = document.createElement('div');
                el.id = 'plot' + i;
                el.style.width = '400px';
                el.style.display = 'inline-block';
                el.setAttribute('plotName', parts[0]);
                el.setAttribute('plotlyPlot', 'true');
                document.getElementById('plots').appendChild(el);
                el = document.getElementById('plot' + i)
            }

            if (!app.ks.hasOwnProperty(parts[0])) {
                app.ks[parts[0]] = {};
                app.ks[parts[0]]['data'] = []
            }
            app.ks[parts[0]]['el'] = el;
            if (!app.ks[parts[0]]['data'].hasOwnProperty(parts[parts.length - 1])) {
                app.ks[parts[0]]['data'][parts[parts.length - 1]] = {
                    x: [],
                    y: [],
                    name: parts[parts.length - 1]
                };
            }
            app.ks[parts[0]]['data'][parts[parts.length - 1]].x = app.ks[parts[0]]['data'][parts[parts.length - 1]].x.concat(
                res[k].x.slice(app.ks[parts[0]]['data'][parts[parts.length - 1]].x.length));
            app.ks[parts[0]]['data'][parts[parts.length - 1]].y = app.ks[parts[0]]['data'][parts[parts.length - 1]].y.concat(
                res[k].y.slice(app.ks[parts[0]]['data'][parts[parts.length - 1]].y.length));
        });

        for (var kk of Object.keys(app.ks)) {
            var data = Object.keys(app.ks[kk]['data']).sort().map(function(ag) {
                return app.ks[kk]['data'][ag];
            });
            Plotly.react(app.ks[kk]['el'], data, {
                title: kk,
                ...layout
            });
        }
    };

    // poll plots
    app.pollLearningPlots = function() {
        $.ajax({
            method: 'POST',
            url: '/plots',
            success: app.drawPlots
        });
    };

    app.exps = {};
    app.experimentsHandler = function() {
        $.ajax({
            method: 'GET',
            url: '/exps',
            success: function(res) {
                app.exps = {};
                $('#experiment option').remove();
                var dopt = $('<option></option>');
                dopt.attr('value', );
                dopt.text('Choose experiment');
                $('#experiment').append(dopt);
                res.forEach(function(r) {
                    app.exps[r['_id']] = r;
                    var opt = $('<option></option>');
                    opt.attr('value', r['_id']);
                    opt.text(r['_id']);
                    $('#experiment').append(opt);
                });
            }
        });
    };

    app.changeChallenge = function(cn, nf, ne, clearPlots) {
        $('#chooseChallenge').val(cn);
        $.ajax({
            method: 'POST',
            url: '/changeChallenge',
            data: {
                agent: $('#agentType').val(),
                searchDepth: $('#searchDepth').val(),
                numFloors: nf,
                numElevators: ne,
                iterPerEp: $('#iterPerEp').val(),
                alpha: $('#alpha').val(),
                epsilon: $('#epsilon').val(),
                gamma: $('#gamma').val(),
                rewardInside: $('#rewardInside').val(),
                rewardOutside: $('#rewardOutside').val(),
                rewardMovement: $('#rewardMovement').val(),
                spawnFactor: $('#spawnFactor').val()
            },
            success: function(res) {
                console.log('Changed challenge successfully.');
                $('#alpha').val(res['alpha']);
                $('#epsilon').val(res['epsilon']);
                $('#gamma').val(res['gamma']);
                $('#trained').text(res['iterations']);
                $('#iterPerEp').val(res['iterPerEp']);
                $('#rewardInside').val(res['rewardInside']);
                $('#rewardOutside').val(res['rewardOutside']);
                $('#rewardMovement').text(res['rewardMovement']);
                $('#spawnFactor').text(res['spawnFactor']);
                if (clearPlots) {
                    setTimeout(() => { $('[plotlyPlot=true]').each((i, el) => el.remove()) }, 500);
                }
            }
        });
        window.location.hash = '#challenge=' + cn + 1;
        window.location.hash = '#challenge=' + cn;
        app.experimentsHandler();
    };

    if (window.location.hash.startsWith('#challenge=')) {
        var chInd = parseInt(window.location.hash.substr('#challenge='.length)) % 3;
        app.changeChallenge(chInd, app.predefChallenges[chInd][0], app.predefChallenges[chInd][1])
    }

    app.hyperparamsHandler = function() {
        var chInd = 1;
        if (window.location.hash.startsWith('#challenge=')) {
            chInd = parseInt(window.location.hash.substr('#challenge='.length));
        }

        app.changeChallenge(chInd, app.predefChallenges[chInd][0], app.predefChallenges[chInd][1])
    };

    $('#alpha').change(app.hyperparamsHandler);
    $('#epsilon').change(app.hyperparamsHandler);
    $('#gamma').change(app.hyperparamsHandler);
    $('#agentType').change(function() {
        var atype = $('#agentType').val();
        if (atype === 'reflex' || atype === 'minimax' || atype === 'expectimax' || atype === 'abprune') {
            $('#searchDepth').show();
            $('#trainMore').hide();
            $('#hyperparams').hide();
        } else {
            $('#searchDepth').hide();
            $('#trainMore').show();
            $('#hyperparams').show();
        }
        app.hyperparamsHandler();
    });
    $('#searchDepth').change(app.hyperparamsHandler);
    $('#rewardInside').change(app.hyperparamsHandler);
    $('#rewardOutside').change(app.hyperparamsHandler);
    $('#rewardMovement').change(app.hyperparamsHandler);
    $('#iterPerEp').change(app.hyperparamsHandler);

    app.logs = {
        sim: [],
        agent: []
    };
    app.logsHandler = function() {
        $.ajax({
            method: 'GET',
            url: '/logs',
            success: function(res) {
                if (res.sim.length > app.logs.sim.length) {
                    app.logs.sim = app.logs.sim.concat(res.sim);
                    $('#simlogs').html(res.sim.join('<br/>'));
                }
                if (res.agent.length > app.logs.agent.length) {
                    app.logs.agent = app.logs.agent.concat(res.agent);
                    $('#agentlogs').html(res.agent.join('<br/>'));
                }
            }
        });
    };
    app.logsTimer = setInterval(app.logsHandler, 500);

    $('#experiment').change(function(e) {
        var exp = $('#experiment option:selected').val();
        if (exp.length > 0) {
            var expdata = app.exps[exp];

            $('#alpha').val(expdata.A);
            $('#epsilon').val(expdata.EPS);
            $('#gamma').val(expdata.G);
            $('#trainIter').val(expdata.T);
            $('#exploreIter').val(expdata.EXP);
            $('#iterPerEp').val(expdata.IPE);
            $('#rewardInside').val(expdata.RI);
            $('#rewardOutside').val(expdata.RO);
            $('#rewardMovement').val(expdata.RM);
            $('#spawnFactor').val(expdata.SF);

            $('#loadModel').show();

            var ch = [expdata.F, expdata.E];
            var chInd = (app.predefChallenges[1][0] === ch[0] && app.predefChallenges[1][1] === ch[1]) ? 1 :
                (app.predefChallenges[2][0] === ch[0] && app.predefChallenges[2][1] === ch[1]) ? 2 :
                (app.predefChallenges[3][0] === ch[0] && app.predefChallenges[3][1] === ch[1]) ? 3 : -1;
            if (chInd > -1) {
                app.changeChallenge(chInd, ch[0], ch[1], false)
            }

            //            app.drawPlots(expdata.results);
        } else {
            $('#loadModel').hide();
        }
    });

    app.experimentsHandler();
    $('#loadModel').hide();
    $('#searchDepth').hide();

    $('#loadModel').click(function() {
        $('#loadModel').hide();
        $.ajax({
            method: 'POST',
            url: '/loadModel',
            data: {
                _id: $('#experiment option:selected').val()
            },
            success: function(res) {
                var exp = $('#experiment option:selected').val();
                var expdata = app.exps[exp];
                var ch = [expdata.F, expdata.E];
                var chInd = (app.predefChallenges[1][0] === ch[0] && app.predefChallenges[1][1] === ch[1]) ? 1 :
                    (app.predefChallenges[2][0] === ch[0] && app.predefChallenges[2][1] === ch[1]) ? 2 :
                    (app.predefChallenges[3][0] === ch[0] && app.predefChallenges[3][1] === ch[1]) ? 3 : -1;
                if (chInd > -1) {
                    app.changeChallenge(chInd, ch[0], ch[1], false)
                }
            }
        });
    });

    app.runTestChallenge = function() {
        var ch = 1;
        if (window.location.hash.startsWith('#challenge=')) {
            ch = parseInt(window.location.hash.substr('#challenge='.length));
        }

        ch += 10;

        app.startChallenge(ch, true, true);
    };

    $("#test").click(function() {
        app.runTestChallenge();
    });

    app.plotPollingTimer = setInterval(app.pollLearningPlots, 1000);

    $("#advanced").change(function() {
        const val = $("#advanced")[0].checked;
        if (val) {
            $("#agentSelection").show();
            $("#hyperparams").show();
            $("#logs").show();
            $("#chooseBestAgentLabel").hide();
        } else {
            $("#agentSelection").hide();
            $("#hyperparams").hide();
            $("#logs").hide();
            $("#chooseBestAgentLabel").show();
        }
    });

    $("#agentSelection").hide();
    $("#hyperparams").hide();
    $("#chooseBestAgentLabel").show();
    $("#bestAgentLoaded").hide();
    $("#logs").hide();

    app.showViableAgents(1);
});