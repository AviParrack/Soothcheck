# 4/15/24

- An excellent post from my good friend Gordon Brander: [<u>Decentralizability</u>](https://subconscious.substack.com/p/decentralizability)

  - It points out that if you decentralize a system from the beginning makes it significantly harder to evolve, and might prevent you from actually making a product with PMF.

  - The right approach is to have minimum viable decentralization to start, and then grow from there.

  - The three components to meet the decentralizable bar:

    - Publish immutable data

    - Use universal identifiers

    - Sign everything with user-controlled keys

- Most ideas get less convincing the closer you look.

  - They start out looking snazzy or polished.

  - But the closer you look, the more you see the gaps in reasoning, or find disconfirming evidence that undermines the simplicity of the idea.

  - In the extreme, something that *looks* superficially great but turns out to be terrible as you look closer is a gilded turd.

  - Concept videos are often like this: exciting… but the more you think about it, the more you realize how unrealistic it is.

  - The more time you spend with the idea, the more its quality regresses to the mean.

    - "Oh yeah, I guess this will be harder than we think."

  - But not all things have this characteristic! Some have the opposite.

  - They look superficially messy, random, rough and ready.

  - But the closer you look, the *more* convincing they become.

  - A sublime jumble.

  - You see that the fundamentals are even stronger than they looked at the beginning.

    - As you dig deeper you have more and more “whoa!” moments, and get even *more* convinced by the idea.

  - This is the mark of a truly rigorous idea.

  - True rigor rewards careful attention and curiosity.

  - When you find a truly rigorous thing, cherish it.

  - These ideas have significant potential.

- Typically we design products top down.

  - We imagine the use cases, then build detailed features to support those use cases.

  - Another way to design is to design for emergence.

    - To design bottom up.

  - In this style of design you think less about the product and more about the system.

  - You look for a small set of principles or rules that efficiently generate fractally interesting / valuable outcomes.

  - Sometimes you can get absolutely massive positive outcomes from only a very small set of very carefully chosen principles.

    - The smallest factoring of rules of the system to allow the richest, most varied output.

  - These kinds of ideas are ones that become even more convincing the closer you look at them.

- LLMs in the computation loop can create more resilience.

  - Last week I riffed on the idea that [<u>systems with humans embedded are more resilient</u>](#za619ugh9jps).

  - Systems with LLMs in the loop also get a similar kind of resilience.

  - Instead of a bare-metal calculation doing precisely what was programmed, you have components in between that can ask questions like:

    - “Is this intermediate output reasonable?”

    - “Does this run differ significantly from previous runs?”

- A good example of an LLM/human in the loop: systems that can be formally checked.

  - E.g. compilers, which will report errors preventing compilation.

  - Or a policy matching logic, which can report configurations that are not allowed by policy.

  - LLMs do a very good job at seeing the error and then proposing tweaks to get the error to go away.

  - This loop can be done without a human in the loop.

  - LLMs are also pretty good at judging the output quality of a thing.

  - This allows a human-in-the-loop style resilience without having to bother the actual human unless you get something viable and good.

- A good tip [<u>via Simon Willison</u>](https://simonwillison.net/2024/Apr/9/a-solid-pattern-to-build-llm-applications/): use the weaker models when iterating on prompts.

  - If you can get something to work in last-gen models (e.g. GPT 3.5 Turbo, or Claude Haiku) then it will definitely work / work better in newer models.

  - But if you do it the reverse, you might find that it’s only viable in the expensive model.

  - The n-1 model will always be significantly cheaper than the frontier models.

- In the past I’ve talked about the power of folksonomies.

  - They allow the collective sifting judgment of swarms of humans to lead to coherent categories, emergently and in an open-ended way.

  - LLMs also should be able to help do folksonomies on steroids.

- What does it mean when something is said to be predictable?

  - That there is an effective model that captures the behavior.

  - Predictable is not some objective free-floating fact.

    - Predictable to *whom*?

  - There are some systems that are predictable to some participants but not to others.

  - When you have relevant knowhow the system becomes more predictable in a way that is difficult to explain to others.

  - An *effective* predictive model is one that can produce accurate predictions… and do so cheaply and quickly enough to actually be useful in practice.

  - A participant who has an effective predictive model in a context has a significant advantage over entities who don’t.

  - If *no one* has an effective predictive model in a context, the entity with the fastest OODA loop will win.

- LLMs allow you to talk to the crystallized intuition of society.

  - LLMs: a deterministic system that outputs vibes

  - LLMs are the apotheosis of the algorithm.

  - LLMs are deterministic, galaxy-scale intuition

  - A crystallized egregore of society.

  - The price mechanism in markets is in some ways the original LLM.

- Google Search is a meta-product.

  - Its quality and usefulness isn’t just a reflection of the effort invested by its builders.

  - The quality and usefulness is also in the quality of the ecosystem it reflects.

  - Even in the early days of Search, if Google engineers stopped working on it, it would still increase in quality as the quality of the web grew.

  - This is the power of meta-products: products that complement an ecosystem.

  - Meta-products can evolve and be open-ended.

  - They reward users experimenting and trying new things.

  - The product itself co-evolves with the emergent uses in the wild.

- At the beginning of the web with slow bandwidth, experimentation was harder / slower.

  - Every link you clicked would take much longer to load, which made users less willing to click.

  - As pipes get faster and latency improves, higher bandwidth experimentation (including video!) gets easier.

  - Faster feedback loops: more experimentation: more users climbing up the ladder of savviness in an open-ended tool.

  - in the future we'll look back at ChatGPT and be like "wow that was super slow and limited!"

- People are willing to invest effort in a product or skill if they expect there to be a return to their effort relatively soon.

  - Things that improve that: seeing other people who are successful at it ("if that rando can do it, so can I!")

  - A first response that is good and intriguing and invites experimentation.

  - ChatGPT has social proof of examples of people posting great results and evangelizing them.

  - People higher on the savviness ladder can help give encouragement to people lower on the ladder. "Use it for X, it will work for you! You can do it!"

  - People who are early adopters, who follow the thought influencers on Twitter, can curate and pass on that knowledge to other people.

- Imagine an ecosystem of software with alternate laws of physics.

  - In this ecosystem, the total value for each user is proportional to the amount of data the user has ever created for *any* use case, and the combinatorial possibility of all of the recipes in the entire ecosystem.

  - This is a meta-product that grows in strength the more that a given user uses the system, but also the more that the ecosystem grows.

  - Over time, each user’s use cases should typically get more and more meaningful and valuable as time goes on.

    - Both as they add in more personal data to the system from any use case

    - But also as the ecosystem grows in size and possibility, even without them.

  - This means that users might start with a long-tail use case, perhaps running a viral AI demo.

  - They might then have a few acute but rare singular use cases.

  - But over time their use would ratchet up and do more and more complex and useful use cases.

  - A smooth ramp up of value with engagement.

- The web's physics set the physics for all consumer software today, the assumption of the same origin policy.

  - The same origin model: it's safe to go to a new place, because the new place starts with nothing.

  - A startup has to start with zero data, a cold start. But the aggregator already has the data, they can start off with a massive unfair advantage.

  - This advantage dominates the "actually a better feature" advantage, and becomes "already has the data".

    - The bar to clear of "already has the data" gets steeper and steeper as the aggregator has more data and the average new app has none.

- The web had a lot of "wow, neat little random thing!" that you liked but won't come back to every day.

  - Remember Stumble Upon?

  - Each individual thing you don’t get that connected to as a user, but overall you get more connected to the whole ecosystem the more you surf.

  - What if instead of surfing the web, it was surfing through experiences?

- Today on Twitter, people doing cool things with LLMs share screenshots, not running links.

  - That’s because there’s no good way to distribute it.

  - If you link to a hosted demo with your own API key, you’ll go bankrupt if it goes viral.

  - If your demo asks users to input their own API key, users won’t do it… it’s a bad idea to give your API key to some random site on the internet.

  - If it’s on your local machine there’s no good way to package it up for others to try with low friction.

  - Something to make that easier as a primary use case could be very useful.

- Imagine a viral meme as an explosion of energy.

  - By default that energy diffuses out into the ecosystem; the viral memes create heat and pressure but to no particular end.

  - But imagine doing the *Three Body Problem* style chain of nuclear reactions plus a solar sail.

  - Each viral explosion doesn't just diffuse, it pushes the overall ecosystem further and further, faster and faster.

- The magnitude of the cold start problem is to some degree an artifact of the current laws of physics of apps.

  - 1\) Making an app is hard.

  - 2\) Distributing it to the right user at the right time is expensive.

  - 3\) When users start with a new app, they start from nothing.

  - 4\) Users have to distrust new experiences until they earn their trust, because the experiences can do anything with data they have access to.

  - 5\) The only viable business model for consumers is ads.

  - But if you were to have an alternate set of physics that changed some or all of these, you might have a significantly less strong cold start problem.

- The aggregators are winning because they already do the algorithmic "comes to you" of content in their feeds.

  - That is, it’s not like the web model of “surf anywhere you want” but rather “stay put and we’ll bring stuff to you.”

  - A few problems with this:

    - This centralizes a god-like power in the aggregator.

    - The emergent optimization function is engagement, not value.

    - It becomes an insatiable gravity well, sucking in absolutely everything around it.

    - The content isn’t turing-complete.

  - That last one is key!

  - As Gordon Brander has put it, [<u>aggregators aren’t open-ended.</u>](https://subconscious.substack.com/p/aggregators-arent-open-ended)

  - This is not *just* a security model limitation.

  - It’s also against the aggregator’s interest.

  - Turing-completeness makes an open-ended system; one that can generate its own requisite variety.

  - If the ecosystem generates more requisite variety than the aggregator, the ecosystem escapes the aggregator.

  - An open-ended ecosystem would disrupt the aggregator.

  - It should be possible to make a system that:

    - Has the “experiences come to you”

    - Is turing complete

    - And isn’t controlled by any one entity.

- How many apps would you sync your Gmail to?

  - Maybe one.

  - Definitely not 20.

  - Syncing your Gmail is a big, scary, deal!

    - Not just leaving one horcrux of yourself in another system, but thousands.

  - If one service was a meta-service, which allowed you to use your emails in lots of contexts safely, that’s probably the one you’d sync with.

- The App model is a poor fit for AI.

  - The consumer app model requires services supportable by advertising.

  - But LLMs are too expensive to be supported by advertising.

  - So you need a subscription… but how many subscriptions will a user pay for?

  - The one-size-fits-all, hardened UX of an app is also a poor fit for the fluid squishiness of LLMs.

  - We need a new physics that is a more native fit for AI.

- The right UX for LLM based experiences is enchanted artifacts.

  - A chat is too squishy.

    - It has a fast, recoverable feedback loop.

      - “No, not like that, like this…”

    - But it’s hard to get any kind of structured / durable output.

  - An app is too hard.

    - If the thing doesn’t work, there’s no way to fix it.

    - The “feedback loop” is so long as to effectively not close into a loop.

  - A file is too static.

    - A file doesn’t *do* anything.

    - Although files do allow a stable schelling point for different applications on the system to coordinate at.

  - An enchanted artifact balances all of these.

    - Can be as squishy as a chat.

    - But as it gets more durable, it gets more solid.

    - It allows a schelling point for various services.

    - But allows them to be active and interactive.

- Imagine: A universe of self-assembling software that orbits around *you*.

- As software becomes more malleable it becomes more ephemeral.

  - Malleable software is easy to get to run once, but becomes hard to manage as you stack more and more of it on top of each other.

  - However, malleable combinations of small hardened building blocks could get the best of both worlds.

  - Like a Lego set!

- One of the most annoying things when putting together a demo is the glue code to other systems.

  - Writing your own error-prone connector logic to other services you’re drawing from.

  - An ecosystem where someone else has likely written the logic, and you can rely on it indirectly, will be more fun to experiment in.

  - More work will go into duct-taping existing components together into novel combinations.

  - Less work will go into users writing the bill ball of duct-tape to be the glue-code to another system.

- The dandelion field is more interesting than the dandelion.

  - Imagine making a more fertile dandelion field, where novel dandelions are more likely to sprout.

  - Someone taking a superficial view might focus in on a single dandelion: “oh look at this snazzy dandelion!”

  - But the dandelion is not interesting, the field is!

  - What’s amazing is not the dandelion, but the stochastic rate of dandelion production, any one of which might turn out to change the game.

- Last week I mentioned a use case of an LLM helping a user reflect and journal, like a counselor might.

  - The user probably would be nervous about sharing that private data with a developer.

  - And developers probably would rather also not be able to see that data and have custody of it.

  - A law of physics where the developer couldn’t see the data would make everyone much more comfortable with that kind of use case and reduce friction.

- In a healthy open network you want even the biggest players to continually have to re-earn their spot.

  - When entities can just rent-collect even without competition, the position will rot and the core of the ecosystem will also rot.

- Ecosystems that grow fast enough are hard for incumbents to tackle.

  - By the time another bigger player notices and decides to tackle them and decides a coordinated strategy to do it, the ecosystem has already grown to be too big for them to wrestle them to the ground.

- A figure/ground inversion in the moment is nothing.

  - Nothing changes, just your perspective.

  - In the limit, however, it’s everything.

  - From that point forward the evolution of the thing is now in a radically different direction.

- Be an ecosystem surfer.

  - Not controlling the waves; riding them intentionally to a great outcome.

  - A surfer has to 1) stay on the board, and 2) use the available wave energy to go as fast as possible.

    - Survive and thrive.

  - When you're surfing the waves, you have to be able to survive even without the wind for some time. And the wind comes and goes in cycles!

- By embracing your constraints you can lean into your superpowers.

- We intuitively think of use cases that we've never seen in a viable form to be inherently non-viable.

  - But actually their viability is contingent on the laws of physics operative in that domain.

- If you solve for privacy, you create a new physics.

  - Everyone else thinks about privacy as a thing to minimize the amount of time and overhead spent on.

  - But privacy is not some random bummer; it’s a fundamental constraint.

  - Instead of wishing it would go away, lean into it as an unavoidable constraint.

  - When you lean into it, it becomes a super power.

  - You can get through the looking glass to a whole universe of things we had previously assumed were impossible, implicitly because of privacy constraints.

  - Lean into the constraint to discover your superpower.

- Where does innovation in the system happen?

  - If at the infrastructure layer you have feedback loop speed problems.

  - If at the higher layer you have coordination / messiness problems.

- A clear distillation of some cybernetic principles from Gordon Brander:

  - “If the error rate is too high, tighten the feedback loop.

  - If the variety is too low, loosen the feedback loop.”

- Getting feedback from someone else is very helpful.

  - They will ask the question from *their* perspective, not yours.

    - It's just someone who is not on the team with you whose ego isn't tied to the current approach.

  - They can be more objective, view the situation from the balcony.

  - An LLM can view anything from the balcony.

    - It has all of humanities' baseline intuition, but no ego.

    - It can critique things without feeling attachment to them.

  - LLMs are average.

    - Which means their biases are evenly spread, not highly concentrated as any individual’s must be.

  - LLMs can be very good at giving cheap feedback on ideas when prompted well.

- When the physics change, a shocking number of your assumptions are wrong.

  - "Things will fall upwards".

  - "Got it!."

  - "No, you don't, everything changes now. Even things you didn't realize were related to gravity!"

  - "... I can throw rocks at passenger airplanes?!"

- AI is the midwit.

  - If the ai could have made a given argument, maybe don’t bother making the argument?

  - Let the AI do the midwit, average things.

  - Humans should lean into their superpowers.

  - Choosing good quests is a moral imperative, to lean into your own particular highest and best use.

  - If you don’t lean into your potential, the chance you do something *great* is low.

- Committees plus subjective criteria leads to conservatism.

  - When you have subjective criteria decided by committee, when any one person says “nah” then it's over.

  - This leads everyone to put forth ideas that are more conservative.

  - Every consensus decision is more about minimizing downside than maximizing upside.

  - Individuals are way more likely than committees to go “I dunno, I’ll have a go!”.

  - If any individual in the swarm succeeds, it lights the way for everyone.

  - How can you empower a stochastic individual in a swarm to find the resonant ideas?

  - How can our software enable more individuals to have a go?

- To steer an ecosystem you need an intuition of the particular emergent game theory.

  - To have lived through it in multiple iterations, to have been an active player, not a passive participant.

  - To have accumulated the knowhow in your bones.

- Ecology is the science of understanding the implications of our actions.

- The fact that everyone can fork is what keeps an open source project trustworthy.

  - Everyone could leave if the owner did something bad, but they don't because the owner doesn’t because if they did everyone would all leave.

- The *option* to extract in an ecosystem is what impedes growth.

  - "We'll just reserve the option to extract from these 5 layers of the future system"

  - But people when deciding to join an ecosystem without a lot of momentum yet implicitly ask themselves, "what's the chance that something happens where I later regret having jumped in?"

  - A system that, by design, resists over-extraction (e.g. where no entity controls the root of trust), will grow significantly faster than alternatives.

- Web3 did decentralization with cryptography.

  - Web 1 did decentralization via contracts and consortia.

  - But contracts and consortia still work as a tactic!

- When you have to convince a lot of different people of a thing, you end up convincing yourself.

  - This can't happen if the idea actually isn't any good. But if you have good answers to all of the questions and concerns you get, you get more convinced of it.

  - By being forced to explain it in ways that lots of people get, you create a resilient understanding.

  - The "I haven't found a question I can't answer" can be a credible signal that the idea actually is good. But there are two things that can make this a less trustworthy signal:

    - 1\) When you know the idea 10x better than anyone else will, so you get superficial questions you can overwhelm them with forethought on; not that you're right, just that you've thought about it more than they're able to in the moment.

    - 2\) When your ego is tied to the thing; if it fails, then you fail. You will do the kayfabe of disconfirming evidence but no more.

- An assertion: a monolithic, unchangeable tool with a world-best AI assistant will be less valuable than a flexible / reconfigurable tool composing an open ecosystem with an OK AI assistant.

- When you own the upside, everything changes.

  - Someone else owns the upside:

    - Being an employee.

    - Renting an apartment.

  - You own the upside:

    - Founding your own company.

    - Doing a hobby project.

  - When you own the upside, there are things that you’d be happy to do that you’d never consider doing if you didn’t.

  - You’re willing to put up with more of a slog, because you feel attachment to the outcome.

  - It feels different when you’re driving your own boat.

- We pay for our own electricity, but we rarely think about it.

  - The cost for an individual joule of energy is tiny; it only matters at larger scales.

  - When we turn on an individual light, we don’t think about how much electricity it’s using.

  - But when we buy a new appliance, or buy an electric car, we think about it.

  - Paying for your own compute would be similar.

- Computation is leverage.

  - It's empowering to pay for your own compute!

  - You own the upside of your own compute, you control how it is applied.

- In some situations there are a number of reasonably good strategies.

  - The only way to mess it up is to thrash between multiple options without committing to any.

  - If you try to find perfection, you’ll end up just thrashing.

  - Pick a good enough one, commit, and then do the next one.

  - Like Tamatoa the purple monster crab talking to Moana: "Pick an eye, babe. I can't - I can't concentrate on what I'm saying if you keep - yep, pick an eye."

- "Overnight success two years in the making"

  - Similar to Picasso's quote: you aren't getting paid for the time to create the thing in the moment, you're getting paid for all of the knowhow and experience and other set-up necessary to create the right thing at the right moment.

  - If all you see is the rising to the occasion in the moment, you'll think it's cheap, or luck. You'll miss that the only reason they were able to rise to the occasion when it presented itself is all of the preparation.

- Things that are illegible to you are moving along a dimension you can’t see.

  - Things are illegible when you don’t have the necessary knowhow to navigate the nuance.

  - Weird things will happen that you cannot fit into your existing mental models in any way.

  - It’s like living in flatland and seeing a sphere: "Wait, how is that circle getting smaller and then bigger? What the heck?

- One of the best ways to influence a slime mold is external pressures.

  - Slime molds can't coordinate internally; they're just an internal roiling cacophony.

  - They have no internal privileged position to lever off, everything is just blooming, buzzing confusion.

- In *Fantasia,* when Mickey enchanted the brooms, he gave them agency.

  - That collective agency quickly ran amok and created an emergent problem that he couldn't clean up or control.

  - Luckily there was that mean old wizard to come back and clean everything up!

- A quick check for epistemic hygiene of a belief system:

  - In this belief system, is doubt generative or evil?

  - If it’s considered evil, watch out, because the belief system is self-totalizing.

  - If it’s considered generative, then even if the belief system is powerful, it is likely because it can absorb and grow from disconfirming evidence.

  - The former is a (dangerous) gilded turd.

  - The latter is a sublime jumble.

  - Another way of putting it: is the belief system open or closed?

- Perfectionism and nihilism are thematically related.

  - Both have a core belief of roughly “anything that’s not perfect isn’t worth doing.”

  - Nihilism adds in “the world isn’t perfect so it’s hopeless”.

  - Both are erroneous reasoning because of a smuggled infinity of perfection being achievable.

  - Once you embrace that perfection isn’t possible, you can lean into making as much good happen as you can.

- Trees don’t make “decisions.”

  - Their “decisions” about how to grow (e.g. which branch to grow the next apple on) are clearly emergent processes.

  - This is obvious because the trees move so slowly (orders of magnitude more slowly than us) and can’t really backtrack from decisions.

  - So it’s easier to see these “decisions” for what they are: probabilistic emergent bets.

  - In animals and humans, we see fast decision procedures and backtrackable decisions.

  - That feels qualitatively different: agency and proper decisions.

  - But what about flocks of birds?

  - They make what look clearly like decisions, but happen emergently.

  - The research of people like Ian Couzin implies that the same logic for decision making in flocks is how our neurons collaborate to make routing decisions.

  - Maybe everything [<u>is more like pond scum</u>](https://subconscious.substack.com/p/pond-brains-and-gpt-4) than we realize.

  - Considering this possibility is depressing, almost nihilist.

  - But look at it another way: the universe is alive and full of agency.

  - The universe is not some passive processes, it is gloriously alive and changeable, and we each are embedded in it.

- When you can feel yourself having a personal paradigm shift induced by a specific person catalyst, it's terrifying.

  - "Do I trust this person to not be trying to harm me?"

  - A personal paradigm shift is a very vulnerable time, a chaotic new stage before you recohere.

  - Like a fiddler crab transitioning to a larger shell as it grows.

  - Someone could really harm you in that transition stage!

- Cancer is hyper-individualism of cells.

  - Cancer is cells that have become selfish.

  - They have lost track of being in harmony with their community and environment.