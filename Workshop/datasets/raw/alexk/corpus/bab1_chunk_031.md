# 4/29/24

- LLMs are inherently bland.

  - If you ask ChatGPT to ask you an interesting question it'll say something super generic like: "What movie or book do you think everyone should read or see, and why?"

  - LLMs only know how to do the average, safe thing, the most regression-to-the-mean thing within the frame you give it.

  - However you can get LLMs to say interesting or piquant things.

  - To do so, you have to give them an interesting *frame*.

  - They’ll still be bland *within that frame*, but the result is still interesting if the frame is interesting.

  - The frame comes from outside the LLM.

    - You use an external force to push it off its balance and into a new catchment area.

  - The default way to do this is to have a human give an interesting frame.

  - But you can get it to happen with a number of external processes.

  - For example, you can set up two LLMs in conversation: one to come up with ideas, and one to critique it like a skeptical user.

    - The ideas start off bland, but with each round of iteration in this loop, they get more and more interesting, leaning into what makes that idea different, accentuating it into something interesting.

  - An LLM on its own cannot be non-bland.

  - But a system with an LLM embedded in it (even if the LLM is most of the mass of the system) can be non-bland.

- LLMs don’t reason, they intuit.

  - With enough scale, this can do an extremely convincing facsimile of reasoning.

    - LLMs appear to be possibly incapable of original reasoning, but so good at hyper-powered fuzzy intuition at scale that they can do a shockingly good facsimile.

    - A nice post on this from my friend Rohit: [<u>https://www.strangeloopcanon.com/p/what-can-llms-never-do</u>](https://www.strangeloopcanon.com/p/what-can-llms-never-do)

  - Humans are *capable* of reason.

  - But the vast, vast majority of the time we do what LLMs do.

    - We use a cached good-enough reasoning answer via hyper-powered fuzzy intuition.

  - Our “System 2” or reasoning center is extraordinarily expensive, and we’d rather not use it very often.

  - Every so often we need to fire it up to calculate a specific reasoned output.

    - Things like tests in academia are designed to *force* you to fire up your System 2 and demonstrate you can do novel reasoning in a context.

    - But in the vast majority of real-world situations, you don’t need to actually do novel reasoning.

    - A good enough cached answer with a bit of fuzzy interpolation is totally fine.

  - Crucially, you don’t have to do the reasoning yourself, you can crib off of what others have done.

  - If everyone around you does a certain task in a certain way and it seems to work well enough for them, why come up with something original? Just do that.

    - Perhaps this is where the human penchant for mimicry comes from.

  - At some point, some human applied reasoning to come up with that hypothesis and then execute it.

  - The idea turned out to be viable and work, so the person did it more times, and more people over time copied them.

    - If it hadn’t worked, they would have never repeated the experiment.

  - The actions we see others around us do are, by and large, fundamentally likely to actually roughly work, otherwise they’d stop doing them.

    - They might “work” for a task that is not what the human intended but is still load-bearing in some other way, as in someone spending their afternoons in front of a slot machine.

  - Across society, a small number of people on any given day need to reason something unique, and then society as a whole can mimic the ideas that survive.

    - Society bootstraps its understanding and ratchets up, adding more good-enough rational moves to our collective repertoire.

  - And now LLMs have come along and can sample all of those existing moves and add them to its own artificial repertoire.

- A useful lens for products: primary vs secondary use cases.

  - I wrote this up in an old [<u>public-but-not-publicized essay</u>](https://docs.google.com/document/d/1E5Fw487KlFCKQwxsb7z2Ln0Rw5onSxZZS_zL-E-Qm-c/edit).

  - A primary use case is one whose expected value for a user exceeds their expected cost.

    - It’s “expected” because it’s based on users priors for how the feature will work, based on:

      - How similar tools have worked

      - What their friends have told them

      - Their own prior uses of this tool

    - “Cost” here means both literal money but also friction, uncertainty, etc.

  - If for a user the expected value is greater than the expected cost, then there is some activation gradient; the tool is viable and over time will diffuse along that gradient and become adopted.

    - The steeper the gradient, the faster the diffusion.

  - The precise expected value and expected cost differs for different users, and can change over time.

    - For example as there’s more word of mouth diffusing information, or the product quality changes.

  - If you don’t have a primary use case to be your wedge, the product is not viable.

  - However it’s not all primary use cases. There are also secondary use cases.

  - A secondary use case is one that is not strong enough to be a primary use case, but is a nice bonus.

    - If a user is already using the tool for a primary use case, and they see out of the corner of their eye a secondary use case that is relevant, they might use it.

  - Secondary use cases can be activated in people’s peripheral vision.

  - Typically the secondary use cases are minor and hard to activate.

  - But sometimes the secondary use cases have a network effect.

    - The more that people use them, the more useful they get.

  - In these cases, even if only a very small number of people use the secondary use case, that trickle makes it higher quality, which pulls in more people: a compounding loop.

  - Some secondary use cases have such a strong network effect that over time they grow to eclipse the primary use case and become their own primary.

    - The original primary use case is like the shade that protects the secondary use case until it grows strong enough on its own.

  - Imagine if you had a tool that had a few good primary use cases, but secondary use cases with hyper-viral dynamics (e.g. multiple interlocking network effects), where it gets better and better with usage at a huge rate.

    - As long as there’s a viable primary use case to get the whole thing going, the overall system could become massive faster than you might think.

- Open-ended tools can have the zombo.com problem.

  - Zombo.com is a joke website from the early internet that had an over-the-top landing page that stated that “You can do anything at all on zombo.com,” but had no actual functionality.

  - Even if your tool is a real one, if users land on it and have no idea what to do with it and thus can’t get it to do anything useful, you could inadvertently have created effectively a zombo.com.

  - This is possible if you can do “everything” but no individual use case particularly well, and with no affordances.

  - With open-ended tools it’s important to have a clear starter use case for users so they can get a sense of what it can do.

  - Google Search and ChatGPT are two open-ended tools.

    - But they are both good enough at so many things, that no matter what you do as a first use case, you’re likely to have a good-enough result that gives you an intuitive sense of what will work and encourages you to try again and use it more.

  - Another approach for an open-ended tool is to have deep links.

    - More savvy friends can send their less-savvy friends deep-links to specific use cases that will work well for what the less-savvy friend is trying to do.

    - The use case is more likely to work for the less-savvy friend to start, which keeps them using the tool and then exploring things it can do.

    - When there is a rich universe of secondary-use cases, the user can quickly get sucked into the secondary-use cases and expand beyond their primary use case.

- Pushing the boundaries intentionally is good.

  - You learn and experiment, and you can judge the upside and the downside of pushing that boundary.

  - Pushing the boundaries *accidentally* is more likely to put you in danger without realizing it.

    - If you don’t realize you’re pushing the boundaries, and erroneously think you’re safe, you’re set up for a nasty surprise.

  - Systems should allow users to push boundaries intentionally but not lull them into doing it accidentally.

- The tyranny of the marginal user is about regression to the mean.

  - If you haven’t read the original piece from my friend Ivan, you should, it’s excellent: [<u>https://nothinghuman.substack.com/p/the-tyranny-of-the-marginal-user</u>](https://nothinghuman.substack.com/p/the-tyranny-of-the-marginal-user)

  - The marginal user represents the gradient towards the most bland thing.

  - If you chase that gradient, you will speed along the gradient towards the most bland, mundane, same-y thing.

    - The maximal regression towards the mean.

  - You arrive at the local minima of the basin of the given domain.

  - A gravity well for all use cases that start anywhere in that basin.

    - E.g. all dating apps regress to a smartphone-optimized experience that emphasizes swiping through pictures to find superficial matches.

    - E.g. all smartphones regress to a 6.7” slab with 3 cameras on the back, 2 on the front, a day of battery life.

- A general pattern: rough in something viable to start and then iteratively improve it.

  - The tendency is often to make a perfect thing before releasing it into the world.

  - But the real world has all kinds of unexpected and non-obvious constraints.

  - There’s a good chance that what you make isn’t viable for some non-obvious reason.

  - Instead, get it to good enough as quickly as possible, get real-world usage, and then iteratively clean up.

    - “Good enough” means “viable for those users who adopt it”

  - If you have a small number of high resilience users, the “good enough” bar might be way below what you think!

  - Especially if the system’s quality is self-bootstrapping, so over time the starter quality improves,

  - Another way of putting it: rough something in that works and then invest more time in proportion to how often it’s used.

  - LLMs are great at roughing something in.

- The same origin model is a one-size-fits-all cage.

  - Very simple to reason about and create, but a poor fit for any given real world situation.

  - You can imagine a system that constructs a bespoke policy cage to allow just what a user actually wants in that situation.

    - Only composed experiences that fit within that cage are allowed to run.

  - Humans could construct the cage themselves by constructing an intricate policy matrix.

    - However, this is time consuming and finicky.

  - LLMs can be used to rough in a “good enough” cage with reasonable assumptions about characteristics almost everyone would agree on.

  - Then from that good enough starting point a human could wrinkle and complicate the rules to better tune them for them.

  - The power of the LLM thus sets not a ceiling on what is possible, but a floor.

  - A human *could* have done all of the work, but that likely would be below the cost/value curve and not viable.

  - The LLM’s starter answer provides a floor that lifts significantly more use cases above that viability threshold.

- Our brains are magic.

  - It’s kind of amazing the kinds of patterns they are able to intuit.

  - I got this one from my four year old daughter.

  - We were leaving gymnastics and heading to dinner at a local restaurant.

  - CarPlay was showing how much time it would take to get home, since we often go straight home after gymnastics.

    - On the screen was the word “Home”, and a map showing how we’d route home.

    - Nothing was said out loud.

  - “That’s funny, the car doesn’t know we’re going to Station Burger!”

  - I was shocked that she could tell that.

    - Either she recognized the word Home on the screen or what the map was showing us.

  - “How did you know that?” I asked.

  - She considered it for a second.

  - “My brain is magic. It tells me things that are true and then I can tell them to other people.”

  - Magic, indeed!

- A pattern: use LLMs for a rough and ready version.

  - A quick and dirty, good enough answer on demand.

  - The more that procedure is depended on, the more you factor out common cases into normal, run-of-the-mill, easy-to-execute code.

  - As time goes on, you fall through to the overflow “rely on the LLM” less and less often.

  - This allows a comprehensively good enough system that can get more efficient over time with use.

- Design a system with humans in the loop.

  - The whole system should be *possible* for humans to do, even if it’s a lot of work.

  - LLMs then provide some good-enough starter ability.

  - The LLMs are the floor and the height of that floor is how good the LLMs are.

  - The higher the floor, the less humans have to do... but humans can always do as much as they want, and reach for the sky.

- A design pattern with humans in the loop: 4-up evolution.

  - This is the MidJourney style feedback pattern I mentioned last week.

  - Provide 4 different options and allow the user to pick the one they like the best.

  - 10 blue links is a kind of version of this, but without a feedback loop that puts you at a new similar decision immediately.

    - If the search results were good enough, you click a link and are satisfied and don’t come back.

    - If the results aren’t good enough, you might refine your query and issue another one and try again.

    - Almost every MidJourney user session presumably has many iteration cycles, whereas Google Search sessions presumably have 1.1 (or less) iteration cycles.

  - It allows the user to be in the driver's seat, but without having to be proactive.

    - It gives them waves to surf to their goals.

  - A Google search with only an I’m feeling lucky would have to hit an impossibly high standard to hit.

    - The likelihood a given query was not good enough is way higher.

    - 10 blue links is way more forgiving, and gives a quality gradient to climb up.

- There's a big difference between "sketching out a a full, reusable app via 4-up evolution" and "I just want to get a thing that works for me in this moment"

  - The former you expect to generalize and be used by others.

  - The latter is OK to be messy and just for you.

- A system's ability to think is its wiggle room.

  - A system that becomes more efficient cannot think, cannot move.

  - All it can do is execute, until it meets something that doesn’t fit, and then it shatters.

- The app model has no wiggle room.

  - Users being able to use LLMs to jury rig solutions doesn’t change anything inside the app model.

  - But outside the app model, being able to jury rig solutions is obviously useful.

- There’s something clear about starting a separate conversation in e.g. ChatGPT

  - You can see exactly what the LLM sees: what’s in the conversation.

  - You can start a fresh conversation if you want fresh context.

  - The new ChatGPT memory feature kind of confuses this feature.

    - Now there are certain things that the LLM chooses to remember.

      - What precisely it chooses to remember, and why, is nondeterministic.

      - If you peek into the memories it’s stored for you, you’ll find all kinds of odd things it thinks you find generally important.

    - This is state that is kind of smooshed across conversations.

      - It’s included in some way for every conversation.

    - In some ways it would be easier to reason about if all the LLM knew was always simply what you can see on the screen in that conversation.

  - The one-conversation model isn’t perfect, either.

  - Conversations are append only; they can’t be modified.

  - This allows you to have an iterative conversation with the LLM. “No, not like that, like this.”

  - But once you get to the right understanding and intermediate state, you want to elide the confusing discussion that it took to get there.

  - It would be better to be able to post-hoc cut out parts of the conversation that turned out to not be useful so it doesn’t get confused by that meandering path to the right intermediate answer.

  - Less like an append-only log, more like a whiteboard.

- Software is expensive to create.

  - That means that software has to be polished, high-fidelity, and one-size-fits-many.

  - But what if software got orders of magnitude cheaper to make?

  - You could have disposable software.

  - Software too cheap to meter.

- The metrics to run an engineered thing vs a living thing are different.

  - An engineered thing is about linear returns, extraction.

  - A living thing is about helping the thing be more alive, to grow as big as it possibly can.

- "Control" and ecosystems don't match well.

  - They're alive! They can be guided but not moved.

  - Your choice as a company is:

    - 1\) a very large, wild garden you don't control, or

    - 2\) a very small, tidy thing you have control over.

  - The garden takes a very long time to grow, which might be longer than the runway you have!

  - A popular request: "I want to make a YouTube".

    - This is a large wild garden that one entity has significant control over.

    - It’s an existence proof of an aggregator that grew as a platform.

    - But it's a distracting black swan; extremely hard to do that from a standstill.

    - It’s best to ignore it as a possibility and instead pick which of the two alternatives are more important to you.

- How should a platform creator know if they should sublimate out functionality done in the ecosystem or not?

  - The platform owner is the closest to a positive-sum perspective on the ecosystem, but also is conflicted because they have a horse in the race with their new functionality.

  - They should ask themselves through the veil of ignorance: "what would the ecosystem prefer? For this to be done by providers or to have an official version in the platform?"

  - The downside for the ecosystem to consider includes things like "pulling the rug out of a company that became successful on the platform" and the overall chilling effect on new entities taking a risk in the ecosystem.

  - There's a gradient from "no platform support" to "promoted partners" to "optional platform functionality" to "required platform functionality"

- One framing of criteria for good code now is "easy to modify in the future".

  - Documentation, tests, etc are in service of that goal.

  - But so is not over-engineering something.

  - Don't build a new code abstraction until you've done it a crappy way at least twice (but ideally three times).

- All APIs are a form of a plugin model, it's a matter of how modular, how much sandboxing there is, how integrated into the host system it is.

  - In a full plugin architecture the host app fully encloses the plugin.

    - Like a eukaryotic cell with a mitochondria at the one extreme.

  - At the other end a single remote API call that you do rarely for a task where you don't care if it fails.

- Often people talk about the mythical 10x engineer.

  - The default frame is that special person who is intrinsically an order of magnitude better than others.

  - Another frame: the person who finds their flow state and maximizes their time in it.

  - People who are in their flow state are 10x more productive.

  - Being in your flow state is not an intrinsic quality; it is contextual.

  - People (and their management) can do a worse or better job at doing work that puts them in their flow state.

- Work on something where it feels like you were cooked up in a test tube precisely for that role.

  - That's how you know it's the highest and best use for you.

- It’s hard to put your heart into something you don’t believe in.

  - When you try to execute on it, you stall, stuck in a chaotic eddy.

  - To execute on a thing you don’t believe in, you have to turn off a part of your brain, the part that cares.

    - To become a zombie, at least in that context.

  - When you’re executing on something you believe in, when it’s in your zone of proximal development, that’s when you can do miracles.

    - To spread your wings and fly.

- Designing your system for extensibility creates an open-ended system that can have surprising upside.

  - Designing a system for extensibility that does not quickly become a quagmire or a security problem is hard!

  - It requires taste and judgment to design it properly. A rare skill!

    - A kind of meta-engineering ability.

  - Sometimes a system that wasn’t designed to be extensible can be extended.

    - E.g. some very popular games have very active game modding communities.

- Ideas that don't require miracles are radically better than ones that do.

  - Most ideas have miracles that we sometimes pretend aren't there (due to kayfabe).

  - But an idea with a full, incremental glide path with no miracles is massively important, even if it looks kind of dinky or messy or otherwise unremarkable.

  - Critically, it’s no actual miracles required… not that you don’t *think* there are miracles.

    - It’s easy to trick yourself into thinking there are fewer miracles than there are.

    - Discovering a miracle is required on a path you’ve tied yourself to is existentially terrifying. It’s much easier to ignore it!

- Minecraft and things like it allow users to incrementally get deeper into it, with no cliffs.

  - Minecraft keeps on piling on new game mechanics, an infinite combinatorial mess of weirdness.

  - It has a whole panda genome dynamic.

  - But that complexity is hidden to you to start, you can slowly explore, from starting at "punching a tree".

  - When you design a lo-fi system, you can make it easy to tinker and experiment and get a fast feedback loop.

- Procedural generation ends up getting at "how would nature have done this"?

  - "What is the most terse way to describe a system that would generate this kind of output?"

  - In game design, even very simple algorithmic systems that are black box, users will interpret as being much more complex than they actually are.

- Network effects dominate everything else.

  - Nothing linear matters next to a compounding force, in the fullness of time.

- Natural selection allows using random noise to propel in a non-random direction.

  - The gradient is implied by survival.

    - You don’t get to pick the gradient beyond that.

    - Things that do not survive are snuffed out of the variation pool.

    - Things that do survive and replicate are more likely to be found in the next time step.

    - This is a fundamental thing that must be true, similar to the inescapability of entropy.

  - Noise that is not useful evaporates.

  - Noise that turns out to be useful is durable and persists.

- Natural selection requires variation to select over.

  - If you have natural selection over the swarm, then the only thing that doesn't work is if no one does anything.

  - As long as lots of people do *something*, there's variation to select over.

  - This is why the Saruman archetype is useful in ecosystems.

    - Any individual Saruman is unlikely to make the successful thing.

    - But a Saruman will do *something* coherent that stands out from the background noise.

      - Something that cuts through the nebulosity to cohere as a distinct thing.

    - And some of those somethings will turn out to be useful and survive.

- Everyone has blind-spots.

  - The only problem is if you erroneously think you don't have any, and have no mechanism for anyone to tell you that.

- There are a number of valuable use cases that are missing not technology but a schelling point.

  - If there were a meta platform that everyone was willing to trust nearly completely to upload their data to, then a number of simple use cases become viable with only a little effort.

    - Especially if everyone knows that everyone knows that.

  - The meta-platform’s killer use case would be the alternate laws of physics of AI.

  - The assistive possibility is a secondary use case.

  - The reason they're killer is because everyone wants them but it's not possible to do in the current laws of physics.

- An ecosystem is intelligent.

  - It’s a different kind of intelligence: collective intelligence.

  - The hotter it runs, the more the ecosystem can do, the more it can think.

  - Swarm / collective intelligence is kind of alien to us ("where do the decisions happen?") but can be significantly more powerful than intelligence with an apparent coherence.

  - Remember, that coherence in a system is inherently somewhat of an illusion (and potentially a costly one!)

- If you can parallel derive something from other sources, it’s not that private.

  - This is the intuitive logic behind k-anonymity.

  - But it also just makes sense: if multiple independent users make some kind of decision, then the more people that do it, the less private the information is.

  - If there’s an action that you know has high intention, then if even a handful of users do it, that's a very good signal that it’s high-intent and potentially generalizable to other users.

  - If lots of users have a given need and seem to not have a good answer to it, that’s the adjacent possible, the frontier of user demand.

  - You can imagine publishing a dashboard to help creators in the ecosystem know what kinds of use cases to invest time in building.

  - A machine could help sort through the adjacent possible, allowing the swarm of creators to fill it in more efficiently.

- We now know it as “the web”.

  - But originally it was “the World Wide Web”: a much longer and more distinctive phrase.

  - Over time, as it became more and more obvious that we were talking about the World Wide Web, it became less and less important to have the signifiers.

    - More and more people could safely elide the “World Wide,” confident that their listener would know what they meant.

  - It’s possible to start with a specific, qualified name, and once it becomes society-scale, the qualifiers fade away.

  - Just “the web”. It’s cleaner.

- Even very small diversions of massive flows have massive outcomes.

  - Page-rank / citation / querying works as a signal if the direct benefit of the action to the person doing it outweighs the aggregate gaming incentive of it.

  - If it's averaged over a large enough population, then the individual action matters less and less, and the incentive to game in non-structural ways goes down, naturally.

  - This averaging over a massive stream making it more resilient to gaming is one of the benefits of massive query streams.

- In a chaotic environment, look back over the last 5 years and pick strategies that would have survived in as many past situations as possible.

  - A sufficiently long time horizon of variance in that context is the best baseline signal to use.

- LLMs are magical duct tape that are made out of crystallized society-scale intuition.

- A pattern when catalyzing an ecosystem.

  - Start out offering an important, but ultimately low-margin component of the system.

  - Charge enough of a margin to make it worth your while.

  - If the ecosystem picks up, others will come in and compete with more efficiency (profitable at lower margin).

  - This is a great thing to happen; it shows the ecosystem has enough momentum to take off, and you can happily cede that business to others and focus on the more differentiated businesses.

- "There's a Saas for that"

  - Kind of amazing how many things you can just pay a subscription fee to.

    - So many different tasks when building a company are possible to have done for you.

  - But also, kind of ridiculous how saturated and late stage we are.

  - In a few conversations I’ve joked about how we’re to the stage where there are "Vertical Saas for funeral parlors"... and someone said, "oh my friend just founded a company that does that!"

- Building conviction is not about convincing others.

  - It's about convincing yourself, by seeking disconfirming evidence.