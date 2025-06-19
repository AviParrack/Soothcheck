# 8/19/24

- My good friend Nick Hobbs has a [<u>brilliant piece</u>](https://docs.google.com/document/d/1_W98tj_Sz6pnpJz3cXNQbxwntkELMHmSUYPy0s1K0Yo/edit) on a new role called AI Designers.

  - He observes that people used to building traditional software are having trouble incorporating LLMs.

  - Traditional software does exactly what you tell it to (which might not be what you *meant*).

  - You can design it precisely and pin it to the wall and it stays perfectly in place.

  - But LLMs are like jello.

    - Squishy computing.

    - Impossible to pin to the wall.

  - A better proxy for LLM apps is not software engineering but game design.

  - Game designers are very familiar with the idea that the core of their software (the game loop) is not something that they control.

  - It is squishy, organic, emergent.

  - You can’t directly control it; you can only poke and prod and evolve it indirectly.

  - My best cheat code for building AI-first software: include an indy game designer on the team.

- Most people aren't programmers.

  - If they want software, they have to hope some programmer somewhere made what they want.

  - That requires that their use case be common enough that there’s a market for the software to encourage someone somewhere to write it.

  - But LLMs give everyone the power of a personal, eager programmer to everyone.

- A pattern we see in LLMs: linear improvement in quality for exponential increases in costs.

  - (Of course, over time we’ve also rapidly improved the efficiency to deliver previously-frontier levels of quality.)

  - Still, if this linear value / compounding cost curve continues, it’s possible to imagine that it’s technologically feasible to create a level of quality in the model that is too expensive to be viable as a business.

  - Supersonic flight is technologically feasible for consumer travel.

    - It’s just not a viable *business*.

- The hard part of software is not writing it.

  - The hard parts are two fold: first, generalizing it, and second, maintaining it.

  - Generalizing it requires bombarding the system with disconfirming evidence.

    - And then tweaking the code to be resilient to the disconfirming evidence.

    - Software in use in the real world is naturally getting bombarded with disconfirming evidence all of the time.

      - But be careful: if the disconfirming evidence is powerful enough, it could kill you or the software.

      - Scary!

    - Programmers also develop an ability to bombard the code with disconfirming evidence in their heads.

      - Imagining the kinds of assumptions that could be violated at runtime and adding in defensive checks.

      - Disconfirming evidence hurts! Most people don’t have the patience or pain tolerance to bombard their things with disconfirming evidence.

  - Maintaining software is the other expensive part.

    - When a family member says "can you make me a quick app?" The reason we shy away from it is not creating it, it's the ongoing working relationship and responsibility.

  - But what if the cost of creating software got so low that it was more often cheaper to regenerate than reuse?

    - Apps that will only be used by one user, or one time, don’t have to be robust.

    - They can be cheap and jury rigged.

    - If you have a small enough audience (one person, one time) then the bar for “good enough” drops to the floor.

    - LLMs can write crappy small software on the cheap.

- Writing code is expensive.

  - It requires an expensive, specialist human.

  - Running code is cheap.

  - LLMs are more expensive than normal code, but can write bespoke code that can be run cheaply.

- The universe of *imaginable* software is way larger than the galaxy of viable software.

  - Viable means a large enough market to justify the fixed cost.

  - Software is expensive to create and distribute, so the market has to be reasonably large to break even and become viable.

  - But if software becomes much cheaper to create a good enough answer for a niche, then do we see more nichification?

  - As the cost of creating good enough software declines, the size of viable niches also declines.

- What is in this whole universe of previously non-viable software?

  - People naively assume that there should be a number of big, obvious use cases.

  - But most big obvious examples are big enough to already be viable in today’s laws of physics.

  - People then erroneously conclude that most software that creates value is viable today.

  - But that’s not necessarily correct!

  - Imagine that most of the things that are not viable are in the long, thick tail of software.

  - This tail of software is “situated software”: hyper bespoke and niche.

  - Let’s imagine, for sake of argument, that the universe of imaginable situated software is orders of magnitude larger in terms of total value for humans than the subset of viable software today.

  - But let’s further imagine that every individual point of value in this universe is smaller than existing viable apps today.

    - If they were larger, then someone would have already made an app for it today!

  - In this case, each individual use case you demonstrated from that situated software universe would look unimpressive, and yet the amount of cumulative value would still be massive.

  - Even if a swarm is massive, if it is made up of uniformly small, long-tail use cases, it won’t look like anything when enumerated as a concrete series of use cases.

  - I believe that today we are missing an invisible and massive long-tail of software.

  - There are massive previously unexplored regions of the universe to explore!

  - This software will be small and situated. Instead of being internally very complex, it will plug into other bits of software around itself, as part of an emergent, flexible whole.

    - Contrast that to software today, which is little expensive, highly designed islands.

- On a scale from hard to soft, we have hardware, firmware, software.

  - But software isn't soft enough today.

  - It's too hard to understand and change.

  - We need soft*er*ware.

  - Credit to Gordon for this frame.

- I love Maggie Appleton’s “[<u>barefoot developers</u>](https://maggieappleton.com/home-cooked-software)” frame.

  - Those are the people [<u>I asserted in the past few weeks are the 9% tinkerers</u>](#73fa7i394pr6).

  - Between the 1% of active programmers, and 90% of people who will passively consume.

  - A [<u>reply to me on Twitter</u>](https://x.com/MarketMarvyMarv/status/1823011764558016855): “I am the 9%!”

- The 1% programmers say messy AI apps for tinkerer programming will always produce messy results.

  - “It can’t ever be as high-quality code as my hand-crafted system”

  - But the 9% tinkerers don’t care, because it’s an infinite relative benefit.

  - Before they couldn’t make a turing-complete thing on their own (zero value) and now they can (some incremental positive value).

  - An infinite increase in relative possibility!

- Engineers can make their own situated software tools to improve their workflows.

  - But for example writers can't do that.

  - The low hanging fruit of “personal tools” is huge.

- For software to interact with other software, it has to agree on a schema.

  - Traditional software is hard and precise; both sides of an API have to be finely machined to fit precisely into the gears on the other side.

  - Agreeing on the schema is a coordination problem.

    - If the two sides don’t coordinate on a precise definition, the gears won’t fit.

    - It’s not that hard for one pair used one time.

    - But as soon as you start introducing more endpoints that all have to agree, or you want to *change* an established schema, it can create significant overhead.

  - A schema general purpose enough for anything is too generic to be used for anything.

  - Schemas require you to reduce fuzzy nuance to precision.

    - But what if the schema you distill is not correct for the domain?

    - The wrong schema is like a straight jacket that’s missing an arm.

    - When creating a schema, you have to be precise enough to be useful today… but also be able to generalize to some degree into the future.

    - A challenging design problem.

  - What if instead you could let schemas emerge organically instead of being engineered?

    - In the past I’ve talked about human-driven patterns using [<u>techniques like folksonomies</u>](#hi3gt0tc43f7).

    - What if you could have LLM-assisted JIT schemas?

    - LLMs interpreting things like OpenAPI specs on two sides and writing bespoke translation code.

      - A massive number of services today document themselves with OpenAPI schemas.

      - Even without OpenAPI specs, LLMs are good at free text and vibes.

    - You could hallucinate the translation code once (at non-zero marginal cost) but then run it cheaply in the future (like other software, effectively zero marginal cost).

- We used to joke that someone who forked Chromium to create a new browser was adopting an elephant.

  - The hard part is not the adoption, the hard part is the care and feeding.

  - This applies, to some degree, to all software.

  - Software is expensive to write, but even more expensive to maintain.

  - But code that was so cheap that it was disposable, you wouldn’t have to worry about maintaining.

- Code written for humans to work with has to fit into a human brain's limited context window.

  - Which requires layers of abstraction; leverage for thinking.

    - A massive accomplishment and benefit... but also leaky, hard to reason about cleanly.

  - LLMs can maintain a quite large context window to patiently sift through.

  - Code written *for* LLMs will have less abstraction, less leverage.

  - Code written *by* LLMs will still use abstraction, because it will mimic code it has seen: code written for humans.

  - But as more code is created that humans don't have to understand, that can be more detailed and less abstracted and still work, the LLMs will learn to code in that way too, slowly.

- In a swarm of bees, each bee does a very simple action.

  - And yet the overall swarm has distinct, large scale, highly complex emergent behaviors.

  - Part of what allows this to happen is [<u>stigmergy</u>](https://en.wikipedia.org/wiki/Stigmergy).

  - The actions of bees accumulates real state in the physical world that other bees can respond to.

    - It could be as simple as one bee being in one location, making it impossible for another bee to be in that precise location at that time.

    - Or it could be accumulating state, e.g. building up beeswax in one part of the hive.

    - This shared state, with each bee modifying it just a little according to some simple program, can lead to amazing outcomes.

    - Not too dissimilar from the [<u>blackboard model of intelligence</u>](https://en.wikipedia.org/wiki/Blackboard_system)

- Imagine: not one big all powerful AI, a swarm of little naive sprites.

  - They're cheap like code. But have some amount of common sense like a living thing.

  - Kind of like a swarm of bees, but where each bee has its own distinctive speciality.

  - You give the swarm an alternate universe without external side effects for them to tinker, experiment, and accumulate state.

    - Everything can be undone, nothing affects anything in the real world, so all tinkering in that pocket is safe.

    - Some external actions would be “safe” (modulo privacy leakage) like “fetch the current weather in Paris”.

    - But some external actions are inherently unsafe, e.g. “buy these plane tickets” or “send the email to your boss telling them what you *really* think.”

    - All external actions would be forbidden in this petri dish.

  - And then you as the overseeing human decide which things they’ve created that you want to pluck over the wall into the real world.

    - Which subset to promote to canon and make real.

  - Semi-automatic software.

    - Software that runs automatically, but always waits for a human LGTM before it does a non-reversible action.

- What will the future be like? One god model that everyone uses for everything, or a swarm of specialist models?

  - My guess is a T-shaped outcome.

  - A small handful of god-like models that can do a good job at everything, but a great job at nothing.

  - And then an ecosystem of smaller, specialized models that have great results on specific domains.

- Toy apps don’t matter if they don’t use real data.

  - The bigger the apps are, the more important it is to use real data, the more grating it gets that they don't do anything real.

- Data tends to accumulate next to other data.

  - Because crossing origin boundaries is hard, all else equal, data tends to accumulate on the side of the boundary that already has more data.

    - More data within a boundary means more option value and potential (the potential value of data rises with something similar to the square of the data).

    - Also the side with more data has more heft, more power to compel the incremental data to come to their side.

  - In a conversation this week someone called it “data gravity” which somehow I had never heard, but I love.

- The same origin model is too coarse.

  - It treats operations *within* an origin as very common, and operations *across* origins as very rare.

    - Within-origin operations are safe and easy.

    - Across-origin operations are dangerous and hard, and the same-origin model just kind of throws up its hands and gives no affordance for them.

      - "If you choose to share information across your domain wall, shrug, you're on your own, that could be fine… or game over level of danger".

  - Why does this happen?

    - Imagine a use case with data that transits over the origin boundary multiple times.

    - Each transit across the boundary is scary / hard.

    - So over time the use case will tend to simplify to remove hops across origin boundaries.

    - The way to resolve a hop is to move all of the necessary data onto one side of the boundary, removing the need for a hop.

    - Which side should the data move to?

    - All else equal, the side that already has more data.

    - Data gravity!

    - This steady but consistent force leads to hyper centralization over time.

  - But if the value of software is the combinatorial possibility of code written by different people being composed into new wholes, then across-origin should be common.

  - A system that leaned into making across-origin compositions easy and safe could remake the laws of physics of software.

- Perhaps the app bundle is being held together more by economics than by ergonomics?

  - Perhaps it’s less about use cases that make sense together for user journeys and more about the ones that have a coherent business model?

- OpenAI’s new advanced voice mode ends most answers with a question.

  - (At least in my experience.)

  - I find this annoying, especially if my question was a straightforward question where I don’t want to follow up.

  - Asking questions at the end of a conversation turn presumes the other person wants to continue the conversation.

  - Humans don't like to leave an obligation unfulfilled, and a dangling question in a conversation is a polite obligation to respond.

    - This is a trick you can use to make conversations you want to continue, continue. For example in messaging with people in dating apps.

  - But LLMs shouldn’t do that, they aren't humans.

  - If the human wants to ask another question, they will!

- People are launching platforms for building things with LLMs faster than people are building useful LLM-native apps.

  - As an industry we learned the “in a gold rush sell pickaxes” lesson, and now everyone is doing it.

  - But maybe it’s still premature?

  - We’re still in the community gardening and experimentation mode.

  - There are lots of products being built on AI.

    - One reason they're quiet is because it works for the initial use case but is hard to extend.

    - If you hold it wrong or use it in the wrong way it doesn't work.

  - Everyone's selling shovels, no one's using them!

  - [<u>https://x.com/voooooogel/status/1797076278329422266</u>](https://x.com/voooooogel/status/1797076278329422266)

- DNS got standardized before anyone realized how monetizable it was to own the phonebook of the internet.

  - Thank god!

- A test for how AI-native your application is:

  - Could it have ever plausibly been viable before LLMs if you had enough capital?

- Measurable beats immeasurable all else equal.

  - If you could choose between a measurable value or an immeasurable one of roughly the same value, everyone would pick measurable, since it’s easier to value consistently to others.

  - A persistent gravitational pull.

  - If the amount of value on either side is roughly the same, that's not a tragedy.

  - But in practice the value on the immeasurable side is an order of magnitude or more larger than the measurable value.

  - The more that everyone expects everything to be measurable, the more that pull becomes, the larger the immeasurable value has to be to overcome it.

  - That's a tragedy.

  - This leads to the phenomena of Serious Business People spending all of their time searching for innovation in the subset of things in the light of easily measurable, missing all of the hugely valuable immeasurable things in the dark.

    - A tragic industry-scale example of the [<u>streetlight fallacy</u>](https://en.wikipedia.org/wiki/Streetlight_effect).

    - In the parable the searcher is drunk.

    - In real life the searcher is a Serious Business Person that is Data Driven and Results Oriented.

- Software in the last decade, by chasing the marginal user, has infantilized users.

  - More and more guardrails to keep users on the safe path we designed for them.

  - But even if users want to go off the path, they can't.

  - The ability of a user to go off the path is a muscle that has to be strengthened for a user to successfully do it.

  - Now a lot of people who could get good at it never started.

  - More people are held hostage to the well-lit paths.

- Every system gets so good at what it does that it destroys itself.

  - In getting ever more optimized, it hollows itself out and makes itself brittle and unable to adapt.

  - Noise and variance is in tension with efficiency.

  - The ability to adapt comes from the noise and variance in the system.

  - You need noise to select over to adapt; the magnitude of noise you have in your system to select over sets your maximum adaptation rate.

  - Innovation and adaptation are a biased selection process over noise.

  - The system gets so good at one thing that it can’t do anything else.

  - When the context changes (and the context, over long enough time horizons, *always* changes), it can’t adapt, and it dies.

- The great big steamshovel isn’t always better than the handheld cordless one.

  - The value of a thing is contextual.

  - And context varies!

- Change comes from challenge.

  - Challenge is uncomfortable.

  - If it weren't uncomfortable, you wouldn't bother to change!

- Keep collecting data until you can model precisely the incremental data you’re collecting.

  - The surprisal is gone when the model is right.

  - This only works if you're actually getting disconfirming evidence.

    - You can get the disconfirming evidence from a truly random sample – both confirming and disconfirming evidence.

    - Another approach is to narrow in on *just* the disconfirming evidence to update your model faster.

  - But be careful: disconfirming evidence *hurts,* and so if you’re applying a selection pressure to what evidence you actually receive and act on, you’re almost certainly filtering out disconfirming evidence.

    - This is especially true in a high-kayfabe environment.

    - Especially an environment with high individual downside and a top-down plan.

    - If you get evidence that shows the top-down plan is wrong, that will cause a *lot* of pain for you!

- Surprise means the model was wrong.

  - Surprise is precious.

  - It’s disconfirming evidence.

  - Use it to make the model stronger and you’re antifragile.

- Consensus implies no surprise.

  - All of the incremental input is already fully captured in the model.

  - Generic, background noise, status quo.

- You are a human, not a cog.

  - Always use your judgment!

  - If any situation wants you to be a cog, not a human, it is an inhuman environment, and might cause you to do inhuman things.

- Alpha comes from the application of calibrated judgment.

  - If you don’t apply judgment you’re just a machine.

  - Machines are consensus.

  - No alpha.

- Most contrarians are wrong, because the market is generally efficient.

  - But if you do everything consensus there is no alpha.

  - So focus on being contrarian in one dimension, the dimension you think you have the highest amount of knowhow in relative to the rest of the ecosystem.

- Novelty you expect is easier to absorb.

  - Unexpected novelty is terrifying.

    - The things you thought were nailed down turn out to be free floating.

    - You have no solid ground, nothing to take for granted.

    - A swirling chaos abyss.

    - Terrifying!

  - How can you proactively create spaces where novelty is a gift, because a) you're expecting it and b) the rules of the space make it unable to kill you?

  - Psychologically safe spaces, where you’ve done the work to earn each other’s non-transactional trust, can be environments where novel, unsafe ideas are easier to absorb.

- If you have a complex idea to communicate, pick two qualities for the argument to have: compelling, succinct, scaled.

  - Compelling - People who hear it find it convincing.

  - Succinct - How many seconds of exposition does it take to make the argument?

  - Scaled - How many people does the one argument convince?

- "This changes everything!" can be exciting, or terrifying, depending on the person and the context.

- When play becomes productized it loses the magic.

  - Play for a point as opposed to for its own sake.

  - As things get lower friction and more optimized they lose their soul.

  - “Don’t you know there’s a game that you aren’t playing to win?”

- Sometimes the most brilliant individual thinkers are bad at bringing people along.

  - To be fair, it's far, far harder to "bring along" an audience on a multi-ply journey.

  - It's not necessarily that they're bad at bringing people along in an absolute sense, it could be that that's the limiting factor for them because the quality of their individual thinking is so strong.

  - Most people are bad at bringing people along… they just don’t have anything particularly challenging to bring people along with, so you don’t notice!

- Weird can be great or terrible.

  - Weird good is great: differentiated, novel.

  - Weird bad is terrible.

  - Weird things are higher beta.

  - Whether it’s great or terrible is about the direction of the perception.

  - That direction is a small term that is multiplied by the weirdness, and it can change quickly because it can be a small term.

- Is it a sweetener or the main dish?

  - The sweetener is the thing that demos well, superficially cool.

  - The main meal is what gives you nutrition to survive.

  - If it's the sweetener and it's charismatic you can fall into improving it to the exclusion of the actual main dish.

    - A charisma trap.

    - Some ideas and problems are charismatic! Intriguing, full of fun challenge, but ultimately a distraction.

  - If you fall into a charisma trap, the sweetener becomes the main meal, and that's empty calories.

    - Looks great, is not healthy.

- People trained only in computer science look at code and see syntax and semantics only.

  - But people with even a little bit of background in some of the “softer” toolkits like history or sociology see whole other dimensions to it.

  - It’s sometimes possible to uncover whole socio techno stories and histories just by closely analyzing the code and the commit history.

  - My good friend Dimitri (who has an [<u>excellent blog</u>](https://whatdimitrilearned.substack.com/)) can do this better than anyone else I know.

    - He can look at a repo and tell you the whole history of it, just like a hunter gatherer can look at a single animal’s footprint and tell you its whole story.

- Once you frame it as a smooth problem (differentiable), it can be optimized and use hill climbing.

  - One of the reasons that identifying a good “self steering metric” can be powerful.

  - For example, for rolling out an ambitious technology: “Maximize absolute amount of usage while minimizing the number of users who have such a bad time they’ll never use it again.”

- Your brain has to be cracked open for ideas about complexity to trickle in.

  - Luckily banging your head against a complex problem for long enough will crack it open!

- Parables don't try to convince you intellectually.

  - The story is charismatic and burrows into your brain, past your intellectual defenses.

  - But now that the seed is there, it might sprout into an intellectual argument or concrete application later, when the situation is right.

- One approach to generating high quality ideas is a “yes, and” stance.

  - This helps find and incorporate disconfirming evidence, and do it in a collaborative, bridge-building way, where the other party feels seen and welcomed.

  - But a “no, but” stance can also identify good ideas in practice.

    - A “no, but” stance throws the conversant off their balance.

    - They are unable to just rehash cached answers that have worked before.

    - That requires them to rederive their beliefs in new ways… possibly in stronger ways than before.

    - You can then absorb those stronger versions of the ideas.

  - Which one you deploy comes down to what personality type you have.

    - Your personality type kind of backs you into a corner of plausible moves that are open to you to use.

  - For example, some people have an almost pathological need to not be disliked.

    - This pushes them into a corner where the only move is “yes, and,” and given that they’re there, they have to learn how to hone that into a highly effective tool.

      - Another thing this kind of personality might do: lean on abstract arguments.

      - Part of this is because the arguments are less likely to be read as aggressive or presumptuous to a person they’re conversing with.

      - And part of it is they’re deeply fearful of someone saying, “no, you’re wrong.”

      - Using abstract arguments is partially a defense mechanism, but also allows them to make arguments that can plausibly apply in a larger set of circumstances.

    - A “no, but” person might start off with a more introverted and more closed personality corner, but once there, they can hone a highly effective tool in that corner.

  - Your personality gives you lemons? Make lemonade!

  - Steelman the situation you are forced into to at least use it to its best effect.

- We all tend to fall into ruts all else equal.

  - If it was good enough in the past, why do something else?

  - In most cases we don’t maximize the quality, we just satisfice.

    - Maximizing takes time.

    - Time is precious!

    - We only maximize the most important things.

  - Imagine a small office with a dozen desks for a half-dozen employees, and no one assigned to any of them.

  - At the beginning, people just kind of sit at desks randomly.

  - But each day afterwards, if someone was unhappy with the desk they were at, they switch to a different one.

  - If they were satisfied with their last desk, they simply stay there next time, too.

  - Sometimes someone swapping desks takes a desk someone else would have used, forcing the other person to choose a new desk, and that can create a cascade.

  - At the beginning, no one has a claim to any particular desk.

  - But after a dozen iterations, if Jeff has sat at a particular desk every time, that de facto becomes his desk, and someone else sitting there would be violating an unspoken constraint.

  - Over time the system settles into a solution where everyone is satisfied enough with it and it doesn’t need to change, so it starts to ossify.

  - A solution not where everyone loves it, but everyone can live with it.

  - If they couldn't’ live with it, they’d make a change or at least make their displeasure known.

  - But if it’s fine, then why make a fuss?

- If you're trying to get a network effect going, the worst thing is to get middling traction on a linear thing that you'd then have to retcon the network effect onto.

  - The network effect is everything.

  - Everything else is a distraction.

  - Linear traction is easy.

  - It's easier to nurse a network effect from the beginning than to bolt one on to a linear thing later.

- When you *choose* to join an organization, it doesn’t make you a better person.

  - You pick the type of organization (for example a club at college) that has the kinds of people who are already more like you than the general population.

    - This allows you to not change who you are, and to just become *more*.

    - A kind of personality echo chamber.

    - Comforting, but not challenging.

  - The way we grow as humans, and the highest performing organizations, are ones with lots of different people thrown into an organization together with no way out.

  - Because there’s no way out, they’re forced to get along, and that challenge helps individuals grow and also the organization to become high performing.

  - Ideally it’s an organization that the participants think is important enough that they can’t just quit as soon as it gets a little hard.

- Horizontal growth is about the fact base of knowledge.

  - Vertical growth is about *how* you think.

    - When you gain a new way of thinking, it’s like learning to sense a new dimension that was invisible to you before.

  - We typically focus on horizontal growth, because it’s easier to grow.

  - But LLMs are quite good at horizontal knowledge, much worse at vertical knowledge.

  - Now that the machines can do the horizontal so much better, maybe humans should focus more effort on vertical development?

  - If you want a great intro to vertical development, check out [<u>https://glazkov.com/adt-primer</u>](https://glazkov.com/adt-primer)

- Calling something “unintended consequences” abdicates the responsibility of thinking through consequences for your actions.

  - "I didn't intend for that to happen!"

  - "OK, but you're the one who pulled the trigger. It's your responsibility to think through the implications of pulling the trigger. Just because you didn't do it doesn't absolve you of the responsibility of doing it."

  - Building on an [<u>insight from Aza Raskin</u>](https://www.wired.com/story/technology-unintended-consequences/).

- If you were right it doesn’t necessarily show you were intelligent.

  - You could have happened to have been right for the wrong reasons!

- One reason Radagast magic isn’t well known is its sphere of influence is very small.

  - It requires huge amounts of trust to work and to acknowledge.

  - Trust is hard to create at a distance, and is orders of magnitude stronger up close.

  - There are Radagasts that can inspire trust at a distance but they’re rare.

- A generative question cocktail party question for someone you just met who told you what they do for a living:

  - "What's the most surprising thing you’ve learned since you took that role?"

  - Gets right at the most distinctive and interesting things, since they found them surprising.

  - A general purpose way for ferreting out the highest alpha information, while asking things that show you are listening and engaged in what they have to say.

- Something is late stage when everything is done for perception first and foremost.

  - The kayfabe overwhelms the ground truth.

  - Everyone is hyper aware of how everyone else perceives everything.

  - A Stanford ish late stage vibe:

    - “I need to be a Google APM or I’m going to *die”*

    - “I started 5 clubs in high school” (but didn’t love any of them)

    - “I’ve got a startup on the side” that they never actually do anything with and then when they get into McKinsey they “shut it down”.

  - When you live in an area with an egregore you don’t resonate with, it’s easy to not engage, to see it for a petty status game.

    - But if you’re fully embedded in the system, that petty status game becomes significantly, existentially real.

    - And if you aren’t yet embedded in the system but are next to it, and it’s an egregore that is roughly aligned with what you value, you’ll over time get sucked into it and fully captured by it.

    - When you are captured by a system and compelled to optimize for perceptions within it you cannot be fully authentic.

- Back in the very early 2000’s, Google was clearly a different type of company.

  - It seemed to do everything differently and better.

    - Optimistic, inspiring, open.

  - They published a list of “Ten Things We Believe to Be True”

  - It seems hard to remember now, given how things have evolved.

  - What companies today feel like Google back then?

#