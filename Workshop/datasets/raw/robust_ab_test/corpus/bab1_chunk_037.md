# 3/18/24

- Software should be soft, not hard.

  - Hard things are efficient but brittle.

    - They’re hard to change, hard to adapt.

    - One-size-fits-all.

  - Software is supposed to be soft.

    - Malleable, adaptable.

  - Software should be like water; fluid, possible to conform to any shape.

  - Apps are hard, like blocks.

  - LLMs are fluid, like sand.

  - Users shouldn’t conform to software.

  - Software should conform to users.

- Software should be a mass noun.

  - Different types of nouns in English: mass nouns and count nouns.

    - One test: you use “fewer” for count nouns, “less” for mass nouns.

  - Count nouns are distinct, individual objects.

  - Mass nouns are a fluid mass.

  - Count nouns: Apples. Diamonds. Apps.

  - Mass nouns: Sand. Water. Happiness…

  - … and software.

  - Software should *flow*.

  - In the immortal words of Bruce Lee: “Be water, my friend.”

- If I asked you to imagine the apotheosis of software, what would you imagine?

  - Almost without question the vision in your head is an app.

    - Perhaps DuoLingo.

    - Maybe Facebook.

  - I think this is a shame!

  - Apps are little non-composable monoliths that are only allowed to exist if Apple says they may exist.

  - Software is so much bigger than that.

  - Software is *alchemy*.

    - It's what allows us to take human agency and extend it beyond ourselves.

    - Software is a language of possibility.

    - It allows coordination and cooperation even between creators who have never met, who never even imagined one another.

    - Software is combinatorial possibility, a fabric of cooperation, emergent leverage for our collective agency as a society.

  - It’s a *tragedy* that we’ve locked it inside the box of an app.

  - AI supercharges everything.

  - In the future we'll either be locked inside of a box with a god AI…

  - …or we’ll finally *escape* the box.

  - People seem to be implicitly assuming the former.

  - With some urgency we must strive for the latter.

  - Don’t imagine software as it exists.

  - Imagine software as it *could* be.

  - As it *should* be.

- You can't retcon safe composability onto a software system.

  - Composability is what allows building blocks to nest within each other to create a whole larger than the sum of its parts.

  - Composability can be a dangerous operation when the components are untrusted.

  - The lower the friction, the more dangerous the composition can be in a naive system.

  - You cannot simply retcon safe, low-friction composability onto a system after the fact.

    - I’ve talked to a number of very smart folks building frameworks for 3P agents to cooperate.

    - I asked them how their security model worked to allow untrusted 3Ps agents to participate.

    - They told me “oh we’ll figure that out later”.

    - To which I replied, “...no you won’t!”

  - The only way to have a system with safe, low-friction composition is for it to have that property from the beginning and then never lose it as you grow and extend the system.

  - The web has maintained a security model of “clicking a link should never be directly dangerous” and “same origin isolation” since the beginning, allowing safe composition.

  - App frameworks don’t have any similar general purpose primitive.

- To escape the box of apps, we’ll need safe, low-friction composition.

  - We’ll need to flip the current privacy paradigm on its head.

  - By doing so we’ll redefine what software means.

  - This will unlock a whole new galaxy of software that was previously inconceivable.

  - We'll shift the power balance back to users.

  - We’ll put the alchemy of software back in the hands of humans.

- The web was enabled by a totally new type of meta-application: the browser.

  - The browser was unlike other applications because it was fundamentally open-ended.

  - For the first decade of the web, “What’s the use case of a browser” had no clear answer.

  - It was like the Matrix: you couldn’t be *told* what the web was, you could only be *shown*.

  - Anything that is open-ended doesn’t have a killer use case.

    - This is why platforms don’t have killer use cases.

  - The web and browsers allowed safe, zero-friction teleportation between, and composition of, untrusted content you’d never seen before.

  - The web enabled whole new classes of software that were previously unimaginable.

- It doesn’t matter if lower layers are decentralized if the top layers are centralized.

  - Decentralization is expensive; it massively increases the coordination cost of adding coherent improvements to the system.

  - Even if the underlying components are decentralized, if the top layer (the way that all users access the system) is heavily centralized, then the system as a whole is effectively centralized.

  - Aggregators aggregate the consumer experience, creating a very small number of extremely empowered linchpins in ecosystems, no matter what the lower layers do.

  - Don't bother decentralizing the layers that don't matter to your problem.

  - Crypto decentralized the "database".

  - But what we need to decentralize is the notion of the app itself.

- When something is one-size-fits-all, you remove choice and agency from users.

  - "Either you use this, or you don't" becomes the only choice.

- In the late-stage aggregator paradigm that we’re in, there's little room for anybody but the aggregators to participate.

  - AI's cost structure threatens to exacerbate that dynamic.

  - The web reduced the floor size and cost of the minimum viable experience, which allowed even small experiences to be viable.

  - The cost structure of AI messes with what we expect.

  - We've come to assume that consumer experiences will be free.

  - But LLMs have a non-trivial marginal cost.

    - Orders of magnitude below human labor.

    - But orders of magnitude above generic compute.

  - By default, this will lead to faster aggregation, as only aggregators can eat the costs by cross-subsidizing from their money makers.

  - The free model leads to one-size-fits-all, hard software.

  - Sometimes there are use cases where you want 30 cents of computation to be put into the task.

  - But sometimes there are important, challenging tasks where you want 30 dollars of computational work to go into the task.

  - With the "everything is free" model you get one-size-fits-all software and you can't choose.

- For abstraction to work, the details have to not matter.

- We often need to capture aspects of the real world in formal, black and white models.

  - These models might be rules, policies, predictions, ontologies, etc.

  - But the real world is fractally complicated.

    - The closer you look, the more nuance you will see, another level of intricateness.

    - This nests fractally and for practical purposes never ends.

  - Each rule costs energy to construct, test, and maintain.

  - The value of a rule is defined by the volume enclosed.

  - The cost is defined by the surface area.

  - But the fractal nesting leads to many orders of magnitude more surface area the closer you look for the same volume.

  - This is effectively the [<u>shoreline paradox</u>](https://en.wikipedia.org/wiki/Coastline_paradox).

  - This is one of the reasons that anyone who has ever tried to fully capture some subset of the real world to any degree of fidelity in a symbolic ontology within some domain invariably gives up roughly 80% of the way through.

  - The costs scale significantly faster than the value.

  - However, at a certain point, you can get away with a fuzzy approximation judged by some other system.

  - In systems made up of humans, that’s often just what a generic employee could reasonably make a call on.

  - But that kind of “reasonableness” approximation used to be hard for computers.

  - LLMs do a great job with straightforward reasonableness approximations.

  - The result is that before you get too many layers of fractal complication deep, you can simply reduce it to asking the LLM.

  - “This cake recipe calls for 5 tablespoons of tabasco sauce. Is that reasonable?” would be the kind of edge case that is very expensive to exhaustively capture in a formal rule system but is easy if you can just ask an LLM.

- With situated software, you’re in a direct relationship with the software.

  - It’s not some black box built by someone else.

  - It’s software that you’ve evolved and grown yourself.

  - You’ve been playing with it, developing your own “theory of mind” to predict what it will do, changed it to behave closer to what you want.

  - Situated software is active software, not passive software.

    - You’re in the loop with the software.

  - This solves a confusing riddle:

    - Why is ChatGPT so compelling, but most LLM-powered UIs are underwhelming or hard to trust?

    - In a ChatGPT conversation, you’re in the loop with the LLM; you can see how it’s responding and tweak your interaction.

    - In traditional UIs you’re on the receiving end of a final prediction with the LLM somewhere deep inside, impossible to access.

- How do new ecosystems get started?

  - First, they have to have a spark at their center.

    - This spark is a new use case that is minimally viable.

    - This means its expected value is greater than the expected cost for some critical mass of users.

  - Second, they grow quickly based on the speed of their network effects.

    - Some network effects have a weak gradient, and some have an incredibly steep one that accelerates with momentum.

      - For example, open systems typically have a much steeper network effect, all else equal.

      - An open system with multiple interacting network effects might be described as having “network effects out the wazoo”

  - Even if a proto-system has a very steep network effect, it doesn’t matter if there isn’t some minimum-viable spark to get it going.

    - This is the “way in” to the ecosystem.

  - A spark has to stand out from the alternatives to get adoption to start.

  - If the alternatives are all very good, then that spark has to be very cleverly selected and invested in.

    - This is because there are a very small number of small sparks that might plausibly stand out, so you must carefully search for and build one.

  - But imagine that some new technology has shown up that has thrown the normal value landscape into a bit of chaos, with everyone trying to figure out what good looks like.

  - This means that finding a spark that stands out is more likely. There are many more plausible “ways in,” any of which might work.

  - The longer your runway, the more open the system, and the more chaotic the alternatives, the more likely that *some* way in will activate, and the less important it is to have *the* way in.

- Folksonomy combines the best of close- and open-ended systems.

  - Folksonomy was a term in the early 2000’s for systems like Flickr’s community tagging system.

    - I imagine I’m one of the only people on earth who have uttered the term in the last few years.

  - The general idea is that the system allows any participant to create a tag; there’s no top-down structure for the ontology.

    - This gives a fully open ended system.

  - But open-ended systems create a chaotic, diffuse baseline.

  - Folksonomies typically work because the UI adds *optional* preferential attachment.

    - That is, users can optionally choose to adopt someone else’s idea.

  - Concretely, this means that when you create a new tag, it first shows you a search result of related tags that already exist, along with how popular they are.

  - The more popular a tag is, the more you’ll be willing to adopt it even if it’s not *exactly* what you had in mind.

    - Adopting a popular tag will help other people find your content.

  - This setup combines the best of open and closed systems.

  - It’s a kind of collaborative sifting sort of good or useful ideas in that context.

  - I imagine embeddings could turbocharge folksonomies if applied well.

- The iPhone did not spring fully formed into the world in 2007 from a stroke of genius.

  - Many on the core development team of the iPhone had worked at General Magic in the 1990’s.

  - At General Magic they’d explored a lot of futuristic personal mobile assistance technologies, but the company failed to successfully productionize it because the tech was premature.

  - But the exploration and discovery they still accumulated as knowhow; they could feel the shape of the problem space (and, importantly, the dead ends to avoid) in their bones.

  - Many years later, the technology had caught up, and the ideas they’d been simmering on for more than a decade finally hit the real world situation that could activate them.

  - The right spark ignited the latent knowhow they had accumulated.

- If it's a tool and you are in control of it, it extends your individual agency.

  - If it has its own agency to do things behind your back, you don’t control it and it’s not a direct extension of your agency.

  - That means that to rely on it you must trust it.

  - If the AI has a memory and agency and access to your tools, then you have to trust it.

  - If it's just an amnesic, neutered endpoint you don't need to worry about it; you can treat it like a tool.

- The app and the aggregator are related phenomena.

  - Apps are harder than websites to distribute.

  - This accentuates aggregation.

  - We’re in the app/agg era.

- An upside of aggregators is convenience, a downside (for the ecosystem) is control.

  - It's possible to get the upside without the downside.

- A frame for AI-created software from James Cham:

  - WYWIWYG - What You Want is What You Get

- Building software is crushingly expensive, so it only makes sense to build polished software at scale today.

  - There's a huge category of software that can be clearly imagined but cannot be built because it is not viable.

  - Vertical Saas is our modern poor man's equivalent of situated software for businesses.

  - It’s one-size-fits-all, but for a smaller niche.

- Writing software is a kind of arcane, unforgiving magic.

  - Wizards have to study very hard to figure out the precise magical incantations to make the machines do their bidding.

  - Now everyone can do that magic!

  - That's got to change *something*!

- Software is bottlenecked by the pain that a human mind can handle.

  - All computer interfaces have to contend with humans having extremely limited swap space in working memory.

- Decentralization is worth it with open-ended ecosystems, not close-ended ones.

  - In close-ended ones (e.g. messaging protocols) the overhead of decentralization just isn’t worth the cost.

- Apple in the 90’s: [<u>Think different. Push the human race forward.</u>](https://putsomethingback.stevejobsarchive.com/think-different-campaign)

  - The juxtaposition of their 1984 ad with their current behavior is striking.

  - Apple today is acting like the RIAA in the 90’s.

- If your arm is tied behind your back, you don't know what your full potential is.

  - Similar to if you are supply-constrained, you don't know the full extent of demand of your product.

- In *The Incredibles*, at the beginning Mr. Incredible advises Dash to fit in and hide his abilities.

  - Not rocking the boat or standing out is of the utmost importance to survive in that context, to be integrated into the broader machine of society.

  - Later, when the family is assembled on the island and fighting for their lives, he tells Dash: “I need you to run *as fast as you can*.”

  - Despite the danger of the situation, Dash clearly feels *joy* in that moment.

  - Being able to push the limits of your special ability, to see what you’re able to do without any constraints, is exhilarating.

- The urgent will take every inch you give it.

  - This will happen even for urgent-not-important things.

  - The mundane toil will take every inch you give it.

  - You need to proactively make space for the important.

- Why do we have deep thoughts in the shower?

  - Because in the shower the urgent thoughts of the next action stop for just for a moment.

  - This allows the important thoughts to have room to breathe and blossom.

  - Another thing that has a similar effect: going on long walks with a dog.

  - Urgent thoughts will always steal the energy from important thoughts.

  - So you need to give the important thoughts a leg up and make some space for them, proactively.

  - Shower thoughts: not just in the shower!.

- An idea in your head is a fragile, ethereal thread.

  - It gets lost, permanently, without continuing active effort to hold onto it.

  - An idea in writing is an acorn: a durable seed that can stay dormant for extended periods of time, but can sprout in the future when the conditions are right.

  - Writing down an idea requires work, to transmute it from a fragile thread into a durable acorn.

  - But when you do it, you transmit the idea into the future with significantly less carrying cost.

  - Instead of the idea evaporating away when you get distracted, it stays as a viable self-contained package of information, with the possibility of sprouting in the future.

- [<u>Write for others but mostly for yourself</u>](https://jack-vanlightly.com/blog/2022/1/25/write-for-others-but-mostly-for-yourself)

  - The primary use case of writing is for you to sharpen your own thinking.

    - To think better.

    - To transmute fragile threads of thoughts into stable acorns.

    - This transmutation requires wrestling with the idea at a deeper level than when it’s a thread.

  - If others find your writing valuable, that’s a bonus.

- Your brain has to let go of all active threads of thought to successfully go to sleep.

  - Active threads of thought tether you to the waking world.

  - If you’re trying to keep track of a thing in your working memory (because it’s not stored in some external system), that’s a thread that will tether you to the waking world.

- If a thing is going to steer itself in the direction you want to go anyway, then don't actively steer it.

  - If you do, you’ll likely oversteer.

  - Just let it drift in the direction you want to go.

- A strong thing can still be in a precarious situation.

  - For example, an intricate copper sculpture perched at the top of a cliff.

  - Strength arises intrinsically but is also contextual.

- Having a strategy that you feel with conviction is liberating.

  - When you have strategic clarity and conviction that is earned, you can set a goal, and employees throughout the org can never doubt that goal is possible or desirable, and they can have fun figuring out how to achieve it.

  - The conviction gives you wiggle room to experiment.

  - You have to actually have earned the conviction from a rigorous analysis that sought disconfirming evidence.

- To grow, you need practice.

  - To practice, you want to first never give up.

    - If you give up, you can’t grow on that dimension.

    - A bar to clear, to satisfice.

  - Secondarily, you want to excel, to grow.

    - If you don’t excel, there’s no stretch to grow from.

    - A bar to maximize.

  - When you excel, you're in your flow state, your zone of proximal development.

    - All of your mental energy goes straight into the laminar flow of your development, not the swirling chaos of everyday distraction.

  - What does ‘excel’ mean?

  - Excel means to push beyond the baseline.

  - The baseline is not what your peers do.

    - Perhaps they've been training for a long time and you just started.

    - It might not be possible to excel compared to that baseline, and you'll give up.

    - Game over.

  - The baseline is what you have previously been able to do.

    - By excelling beyond your own personal baseline, you are growing.

    - And you're growing in a way that is achievable and unlikely to be game over.

  - Compare yourself not to your peers, but to former versions of yourself.

  - At the beginning you might be far behind your peers. But if you keep growing, before you know it you'll accumulate enough progress to be in the running.

- An idea from cybernetics: error is what drives the system.

  - Error is neither good nor bad.

  - Error is the variance that gives you feedback signal to improve.

  - Without error, you cannot grow or change.

  - The messiness is where the aliveness comes from.

- A friend, musing on my bits and bobs process: “You eat all of the parts of the intellectual animal!”

- Perfectionism is about clinging to the illusion of complete control.

  - But it's just an illusion.

  - And in uncertainty, an extraordinarily expensive one, with super-linear costs.

- The fundamental rule of good design: make it look intentional.

  - "I made it to look like this, because I was in control." vs "I dunno, this is what it ended up looking like".

  - Design is about a credible demonstration of control.

- Thriving companies feel alive, they have unexpected upside.

  - Like any living thing, they escape the control of their creators.

- Being vulnerable as a leader is the opposite of insecurity.

  - Looking weak and *being* weak are often orthogonal.

  - Someone who looks macho could be very insecure.

- It’s easy to get into a situation where you think you’re being tough but you’re actually punching yourself in the face.

- The figure ground inversion doesn't change anything in the moment.

  - Instead, it changes which direction you grow in from then on..

  - That's what makes it a subtle but profound shift.

  - As a reminder, an example figure/ground inversion: “United States” going from a plural noun before the Civil War (emphasizing the states) to a singular noun after the Civil War (emphasizing the union).

- When you're put in a no-win situation, swing for home runs.

  - Because you'll be dead soon anyway by default, so you might as well have some upside!

- If you're doing 5-ply thinking but you're getting the 1st ply wrong, you are wrong.

  - Multi-ply thinking is dangerous and hard, because if you get any of the plys wrong, all of the later plys are wrong, too.

  - Multi-ply thinking is leverage.

  - Like any levered thing, if anywhere along the lever breaks, the whole thing topples.

- If you optimize for agility above all else, you'll focus on urgency, not what's important.

  - 1-ply thinking.

  - Goes fast in the short-term, slowly in the long-term.

- Lateral thinking with weathered technology: ordinary components, extraordinary results.

- Cultural expectations in a company coevolve.

  - People do things that they've seen others do, and that have seemed to have worked for others.

  - Social systems of norms are alive, and shockingly resilient.

  - An example: Google’s peer bonus program.

    - It allowed any employee to give a \$200 bonus to any other employee for just about any reason.

    - There were a few limitations:

      - No more than 12 a month.

      - The recipient's manager got an opportunity to veto anything inappropriate.

      - You couldn’t peer bonus someone who recently peer bonused you.

    - Every year, some enterprising new APM would figure out a “triangle peer bonus” to create rings of peer bonuses that skirted the rules.

    - But whenever an elder APM heard about it, they’d say “that’s not cool man,” and the new APM would cut it out, no formal rules necessary.

    - With that little bit of social energy, the thing went from a “I’m cleverly exploiting a loophole” to “I’m undermining a system everyone values.”

- Large systems have to invest most of their energy into surviving.

  - In biology, it’s called basal metabolic rate (basic body processes) vs field metabolic rate (doing useful things).

  - Large systems grow complex internal processes that require a lot of energy expenditure.

  - In rockets, every pound of rocket fuel to lift a payload needs additional rocket fuel to lift the net new fuel (luckily, just a smidge less), ad infinitum.

  - In organizations of people, every additional person requires more coordination, which takes more people power to handle.

  - The largest organizations spend the vast majority of their overall effort just on dealing with internal organizational dynamics, not with producing externally-visible value.

- When someone underestimates you you have an asymmetric move against them.

  - You can do an end run around their expectations.

  - This is only true if the person who misunderstands you can’t knock you out of the game.

  - If your manager underestimates you, that’s more likely to be a problem!

- Humans hate letting down a commitment they made to others, explicit or implicit.

  - Especially the more people who are aware of the commitment or see it not be met.

  - Because for people to trust your word you need credibility.

  - Credibility is the credit you’ve earned from the precedent of always meeting your commitments.

  - Your word, your integrity, is your power.

  - Shame is the powerful emotion baked into our firmware to make us follow through on commitments.

  - Human arrangements implicitly take advantage of this fundamental brain circuit:

    - Accountability partners in workouts.

    - A large guest list at weddings.

    - Someone twisting your arm to take an action item in front of the boss.

  - When people follow through on commitments, society is significantly more effective.

  - People who are low conscientiousness will sometimes not perceive a commitment where others do.

  - People who are high conscientiousness will sometimes perceive a commitment even where others don’t.

    - This can lead to a shame spiral.

    - That loop can propel you forward to be constantly improving and collaborative… while being toxic for you as a human.

- People don't use products because of a bonus feature.

  - Only a primary use case sells a product.

  - A bonus doesn't sell a product.

    - It's a secondary use case.

    - It only makes a thing that people already want, want more robustly.

  - A secondary use case is an accelerator, a primary use case is an enabler.

  - A secondary use case has to not increase the cost of the primary use case.

- The most resilient experiences built using AI will get better with AI, but work even without it.

  - AI as a bonus.

  - If the experience is entirely powered by AI, then if it fails in a given situation then there's no recourse.

  - If it’s an experience that allows users to wire together things, and the AI helps them do it, and if the AI fails, a savvy user can open up the side panel and wire it together themselves.

- When a user adopts your product they’re saying they think it’s special.

  - That is, it’s worth exerting energy to adopt.

  - When some meaningful subset of people think your product is special then you have PMF.

  - If someone who has a vested interest in the thing (e.g. investor, boss, someone whose day job is working on it) says it’s special you can’t tell if it’s actually special.

  - The mark of it being special is entirely people without a vested interest taking actions that demonstrate they think it’s special.

  - Everyone thinks their thing is more special than it actually is.

- Doing a 1 year strategy before a 5 year strategy is harder than the reverse.

  - (This is only true for post-PMF companies)

  - A 1-year strategy needs to wrestle with quite a lot of rich detail about specifics.

  - A 5-year strategy is allowed to (and should!) elide a lot of details to focus on the core narrative arcs.

  - If you do a 1-year before a 5-year, you need to somehow retcon all of the messy real stuff into something that sounds like it’s intentional, and it has to stand on its own.

  - If you do a 5-year strategy first, and you do it rigorously, then you can focus on the most important throughlines which is much easier to rationalize.

    - The future obscures lots of details, so you *shouldn’t* have unnecessary details or precision in your very long-term strategy.

    - A lot of day-to-day trade offs simply evaporate over sufficiently long time horizons.

  - Then, when you do the 1-year strategy, you don’t have to make it make sense on its own.

  - You can simply point at the 5-year strategy, which does make sense, and show “this is our incremental step from where we are today towards where we’re going”.

  - Many orders of magnitude easier!

- In a healthy org, leaders up and down the stack should have some conditions where they can say no.

  - Whenever there is a big power differential in a discussion, the person with the most power by default wins.

  - This can quickly mean that bad decisions get made because the local relevant context is only known by the significantly less powerful person.

  - Knowhow is difficult to transmit, which means someone with relevant intuition on why something is more costly than it looks might not be able to succinctly communicate it.

  - To the very powerful leader, it will look like the underling doesn’t have a good reason, but actually it’s just that the reason is hard to communicate.

  - This is one of the reasons that senior leaders shouldn’t get involved in very small details.

    - Because when they do it will be very hard for the actual right answer to be decided due to the power differential.

    - And everything the very senior leader *might* look at has to be handled defensively, in case they do in the future.

- In a large company, teams might have low or high autonomy in processes or strategy.

  - Low Process Autonomy / Low Strategy Autonomy

    - A command-and control style; great when quality is of the utmost importance and the conditions to navigate don’t change much.

    - Everything feels like a sub-component of the larger thing; teams can coordinate easily because everyone has the same process.

  - High Process Autonomy / High Strategy Autonomy

    - A chaotic but resilient soup of different orgs that behave like mini-companies, not part of a large whole.

  - Low Process Autonomy / High Strategy Autonomy

    - A swarm of sports cars.

    - Teams have considerable autonomy to make decisions themselves, but can coordinate with other cars easily because of the shared processes.

  - High Process Autonomy / Low Strategy Autonomy

    - The worst option.

    - Teams can’t make high-quality situated decisions for themselves, but also teams work in totally different ways so they can’t coordinate.

  - Process is more important to be robustly tolerable than precariously optimal.

    - There are a lot of different processes that all work reasonably well.

    - What’s important is not which process you pick, just that everyone picks a compatible one.

    - When driving, it doesn’t matter if you drive on the right or left. What matters is that everyone in your area picks the same one.

- If it's in their blindspot, it's not that they disagree with it, it's that they literally cannot see it to engage with it.

- Acorns aren’t just “projects”

  - They also include things like:

    - Riffs

    - Documents

    - Relationships

  - The more conditions they blossom in (where a reader has an "Oh!" moment), the more robust they are, the more important they are to invest in making into a durable format that can be shared easily.

- A generic cold-reading statement that applies to basically anyone in any organization:

  - "I mean I'm optimistic the official plan will work, but... there sure are a lot of challenges!"

  - A statement that is safe, and also always true, and somewhat subversive.

  - It allows the other person to glom on the additional challenges they see too.

  - When used in a private, candid environment, it’s a magnet for disconfirming evidence, giving others permission to point out challenges they see.

- A gossip technique in internal orgs that relies on canaries.

  - No one wants to talk out of school about a thing they were told in confidence or know officially.

  - But when they have just a theory of a thing they will happily share it with others.

  - Once they're officially briefed, they can no longer share it, and also can't disconfirm it.

  - So they'll go silent and try to move on from gossiping on that topic if it comes up.

  - Their refusal to continue gossiping about a previously gossiped topic is a canary it's about to come true.

- In rudderless large organizations, ideas just have to stand out compared to entropy.

  - Someone working with intention can likely find little ripples to accentuate into larger waves to ride, because there isn’t a strong opposing force, just randomness.

  - A very different situation: the bottom-up entropy of any large organization… but with a very empowered senior lead with a specific, and perhaps on some topic wrong, perspective.

  - Those are way, way harder to make good things happen without convincing that key person.

  - You can’t find waves to surf to good outcomes, because they all emanate from that one person’s perspective.

- When you can see a path to something great, you get an almost unstoppable energy.

  - When you don’t see a path to something great, you’re an order of magnitude less motivated.

  - Large organizations, by their very nature, have constrained paths to greatness.

    - The naturally-occurring eddies of interpersonal coordination slow down any clear shots.

  - That means that in large organizations, there are few *obvious* paths to greatness, and it’s harder to get intrinsically excited and in your unstoppable flow state.

  - That’s not to say that the person can’t do it, just that they can’t see their potential in that context.

- I was in a collaborative brainstorm recently that used Jamboard (RIP!)

  - The board of virtual stickies kept remarkably organized, emergently.

  - I noticed the implied algorithm we were all following was simple:

    - 1\) Nudge around things to have less overlap.

      - A kind of simple force-directed layout

    - 2\) Move things where their new neighborhood is *clearly* a closer semantic fit than their current one

  - It has a kind of simulated annealing kind of behavior.

    - At the beginning it jumped around a lot, but later it became mostly small tweaks.

  - I bet you could automate the second part of the algorithm to some degree with embeddings.