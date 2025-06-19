# 6/24/24

- A proper Copernican Shift massively clarifies things.

  - Before the shift, you have to keep on glomming on complications to make the old model fit with reality.

    - Things like epicycles and other curlicues in your model.

    - The only way to make the model fit with observed reality is to heap more and more complexity on it.

    - Climbing the wrong hill at greater and greater expense.

  - But when you flip to the correct model, a ton of complexity falls away.

    - A cleaner model of the world as it actually exists.

  - How much of the complexity of software today are epicycles and curlicues on top of laws of physics that don’t match how users actually want services to work?

- LLMs don’t do a good job with negative space.

  - A friend was trying to generate a picture for a game he was a dungeon master for.

    - He asked a generative AI to make a picture of a secret lair in a swamp.

    - The picture that came back was perfect, except there was a small sign that said “Secret Lair”.

    - He tried again, appending to his prompt: “There should be no signs whatsoever”.

    - The next image had a *larger* sign.

    - This continued, escalating until the generated image had a flashing marquee sign pointing to the secret lair.

  - This pattern will be familiar to anyone who has wrestled with generative image models to remove some detail.

  - All of the training data has descriptions of images as they actually are.

    - In that case, why would you describe what’s *not* in the image? You can just describe what *is* in the image.

  - But a description of an existing picture, and a description to *create* a picture are different.

    - In the former, you would never use a negative word like “without”.

    - In the latter, you might, if the baseline understanding of the artist might guess that a certain detail should be included.

  - The generative model isn’t used to seeing negative/”without” words, so they’re effectively background noise to it that it ignores.

  - As you get increasingly emphatic about what to remove, it still doesn’t sense the negative word, so all it sees is “wow, he seems really emphatic about this secret lair sign, I guess it should be really emphasized!”

- Another puzzle of LLMs: they’re surprisingly bad at generating very large legal JSON blobs.

  - They’ll often miss a comma or a } or \].

  - This breaks our mental model; they’re so good at generating things that match even subtle patterns, and this is something a simple pushdown automata could handle!

  - LLMs are not doing reasoning or computation, they're doing extremely good vibes matching of things they've come across before.

  - There's so much JSON in the training, that especially for small blobs of JSON there's tons of examples of every permutation of nesting of objects and arrays, and so its intuition / vibe matching is very resilient.

  - But the larger the JSON blob gets and the deeper the nesting, the fewer direct examples of that exact nesting structure there is.

    - This must be so for structural reasons, as with each step the combinatorial space grows.

    - *Something something* [<u>Assembly Theory</u>](https://www.nature.com/articles/s41586-023-06600-9).

  - LLMs have to have their attention layer tuned from examples to know which characters to attend to.

  - They aren't "counting" the parentheses, they're pattern matching based on things their attention mechanism tells them are relevant for situations like this... and those are tuned based on the stuff it's seen in training data.

  - So we get confused that *they* get confused because they aren't doing basic computation, they're doing vibes matching on a mind-numbingly large dataset that gives them resilient coverage of any smallish JSON shape.

- The more tests you have on your codebase, the more that LLM based autopilots can detect its own errors as it proposes changes.

  - A passing test doesn't tell you that it definitely works correctly.

    - Maybe it fails in a way that you haven’t encountered before or captured in a test.

  - A failing test (or a failing compile) almost certainly means something is broken.

  - A generally useful pattern, especially if you’ll have more automated assistance: smoke tests in every random direction to make it more likely a non-viable thing is detected.

  - Tests get more important... and also LLMs make them considerably easier to write.

- When LLMs talk to each other in a loop, they can do smarter things than they could do alone.

  - For example, have one agent generate ad copy, and another agent critique it as a cynical consumer. Then repeat.

    - These can generate significantly better results than a single-shot generation.

  - Where is the net new intelligence?

  - The intelligence is the loop!

- Within a niche there is an easy-to-execute plan, but the returns are capped.

  - Niches are easier to get a toehold and then grow that into a durable (but small) position.

  - Niches are defensible, small enough that once a player is established it's very hard for a competitor to muscle in.

    - Like a little protected cave.

  - Niches are less vulnerable to execution risk or competition within the niche, more vulnerable to structural risk that invalidates the entire niche and its whole neighborhood.

- Over-optimization is pushing deeper into a niche.

  - A niche is a safe, protective, defensible, valuable region.

  - But a niche is also a rut.

    - A rut can trap you in that region, which can be deadly if the entire neighborhood becomes non-viable.

  - A rut is structure that protects and makes you more efficient, but also traps you.

  - A too tight feedback loop leads to homogenous over-optimized things that are prone to catastrophic population collapse.

  - A too-tight feedback loop leads to over-optimization.

- A configured spreadsheet for your domain including best practices as guardrails is pretty useful, actually!

  - This is one of the reasons the vertical saas niche techniques are so effective.

  - Own a niche and then set defaults for everyone that are pretty good and put everyone's mind at ease.

    - A customer: "I don't know what I'm doing but I know this works!"

  - When the customers are in different geographical niches they don't have to worry about competing as much with other customers of the platform doing basically the same things, following the path of least resistance of the guardrails of the tool.

  - Although this nichification creates monoculture.

    - Lots of geographically distributed companies who are all executing the same playbook; if that playbook is invalidated it could kill all of the players executing it.

- Your plans should change by default in proportion to the amount of surprisal.

  - “The thing we thought would happen did happen”: Some uncertainty has resolved but in a way that doesn’t change your thinking.

- Being the referee requires you to not be in the game.

- Minecraft has a smooth path of discovery from starting out punching a tree to building a computer out of redstone.

  - A self-propelling path of discovery, creation, play.

  - Not serious or goal oriented or over-optimizing.

  - Here are your things, and here’s a crafting table to create more things that can be used to make more things, a combinatorial explosion of possibility that smoothly reveals itself to you.

  - Never feeling taught, feeling a sense of discovery.

  - A concept that would be cool: Minecraft for your data.

- Back in the day Flash was magic because it put *what* you wanted to do as an artist/engineer/tinkerer ahead of *how* to do it.

  - It allowed a kind of bottom-up, uncomplicated exploration where the complexity revealed itself as you got more knowledgeable.

  - Similar to the Minecraft punch-a-tree-to-redstone-computer smooth iterative path of discovery.

  - Who can reclaim that magic?

- The quality of a ranking system comes from multiple components.

  - Only one portion is the algorithm itself.

    - This is the component that is *built*.

    - It has a linear impact on quality.

  - The other part is the swarm of data the algorithm operates over.

    - For example, aggregating and distilling insights out of the combined behavior of the whole swarm of users.

    - This is the part that is *grown*.

    - It can have a super-linear impact on quality.

  - Even very simple algorithms, when distilling the insights of a massive swarm, can have great results.

  - If you don’t harness the intelligence of the swarm in your ranking system, you’re cutting out what should be the dominant source of intelligence in your system.

- An ideal system would benefit from the insights of a swarm of users.

  - If there is a lot of overlap of queries from users, then even small bits of engagement and signal can be used to improve the quality for large classes of users.

  - The first time an automatically-generated answer is shown to a user, it’s not known to be very good.

    - LLMs return answers that society cached in oft-repeated utterances.

    - But until a human says "yeah, that looks OK" or even better "yup, this is good" then you don't know if it's appropriate to cache and return this answer automatically, quickly, to future questions from other users.

    - But the more users who see that result and don’t barf, the more likely it is to be good.

    - Even a small number of highly motivated users interacting with the result in a high-intent way (e.g. pinning it) is a great sign it’s useful.

  - How can you have humans in the loop implicitly with small bits of signal that stand out from the background noise

    - Humans are a form of ground truth, pinning down what's valuable / useful.

    - If you have humans in the loop and the signal can pop out of the general noise, you get a very powerful, self-ratcheting quality system.

    - Even if only a small proportion of users lightly interact with only a small number of things, at scale that's still enormous amounts of signal--so long as the other interactions from users that don't line up with quality are uncorrelated.

    - A pattern where to any specific individual it looks like noise, but if it's consistent, at the level of the population it pops out clearly when you average.

      - A thing that if you look at the swarm as a whole it looks just like noise, but the consistency of alignment creates coherence.

  - That signal, at large enough scale, could help ratchet up quality significantly.

- "I want to eat more vegetables."

  - "Ok... but do you though? Or do you *want* to want to eat vegetables?"

- Meaning is slow.

  - Optimization is fast.

  - Fast twitch feedback loops are about optimization not creating meaning.

- The app model can't do speculative assistance.

  - Speculative assistance is necessary to do anything exploratory, where you don't know what the answer of the service will be before you do it.

  - But in the same-origin paradigm, once you reach out to the 3P service, that service could do whatever they want, on a technical level, with that data.

  - That data could leak out in unexpected ways.

    - For example, “What would a family trip to Tokyo in the summer look like?” could turn into getting unexpected emails from Delta encouraging you to try their co-marketing deal at sushi restaurants.

  - This means the app model is an awkward fit for an AI-native speculative assistance.

- To OS assistance layers, apps are dumb receptacles of functionality.

  - Dead, not alive. Not agents.

  - There to be called on by a higher intelligence (the user, or the OS) but to not speak unless spoken to.

- When the feedback loop is long and expensive you’ll cling to any signal you have.

  - You’ll lean into superstitions.

  - Including things like taking offhand comments from execs as extremely important constraints to optimize for.

  - These kinds of superstitions can be self-strengthening.

    - If everyone has the same long feedback loop, everyone will cling to any signal.

    - If everyone else is taking the signal seriously, and you don’t, you’ll stick your neck out even further.

      - How much your neck is sticking out is relative to what everyone else is doing.

    - Why risk it? Why not just go with the thing everyone else is taking seriously, just in case it is real?

  - This can give a surprisingly durable dynamic that no one actually thinks is that important.

- If you don’t make a junk drawer you’ll get less junk.

  - Junk that cannot be stashed out of sight is less likely to get generated and kept.

  - The amount of junk is not some static distribution; it coevolves with how many places there are to stash it.

  - An equilibrium of misery.

- A simple recipe to be insufferable.

  - Liberally insert “simply“ and “well, actually” into your conversations.

- Any given useful variance is lost in the noise for the individual.

  - Useful == gave an edge to survive.

  - There’s so much variance that the useful bits are swamped.

  - But at the level of the swarm, you average the population together and the useful variance is what stands out.

  - That's a consistent edge imparted by the ground truth of the universe.

  - That strong consistency of a subset is what makes it pops at the level of the population.

  - Usefulness gives a consistent edge, and that edge is determined by what actually ends up working in practice in the ground truth.

- Language in general is a folksonomy.

  - People use a word that others understand (that stands out from the background noise of other words) and the more people that understand that word to mean that, the more effective the word gets for that use, and the harder it gets to change.

  - Random variations, based on emergent schelling points, that then get accentuated and strengthened and turn into something durable and real.

  - An emergent evolutionary process, swarms of people attempting to communicate leading to a flood fill of possibility, a coherent, branching edifice of meaning.

- It's kind of bonkers how much signal is encoded in cooccurrences of words!

  - But there's a clear alignment of sentences that are useful in the world.

  - A weak but consistent alignment based on the ground truth of useful utterances in the real world, which allows that inherent structure to pop out at scale.

  - People say things in the real world to make things happen, which means those utterances have to align with ground truth usefulness, which means they have an implicit structure to extract meaning out of.

- It's not that economics is wrong as a lens, it's that it's just a lens.

  - If you think it's the only thing that matters then you'll be wrong.

  - Any lens held too tightly is wrong.

  - Someone whose whole identity is tied up with a specific lens will overapply that lens.

  - Taking it off will feel like cutting out their eyes.

- LLM: A mind of a toddler who has also read every book in the world.

  - It's confusing to us, we don't have any points of reference for it.

  - We get tricked by an erroneous mental model of what it can and cannot do.

- A vibe I like: "Conjure up a UI"

  - Something magical, imprecise, and creative.

- Imagine a massive mountain everyone can see.

  - Everyone can see "There's no way up the mountain!"

  - The vast, vast majority of people who try to make it up the mountain will die.

  - But the lucky few that make it through won't have realized how hard it was when they started!

  - The person who makes it up the mountain (if anyone ever does) will say "I didn't realize it was impossible, I just went up the mountain. If I would have known how impossible it was I would have never tried."

  - This is one reason Silicon Valley can find new peaks: a steady stream of bright eyed / bushy tailed young founders who don't know what they don't know.

  - The overall ecosystem can use that energy to sandblast at problem domains and find any hidden paths that exist.

  - Every so often there really is a hidden, specific path up a dangerous mountain.

- The laws of physics of a system provide the underlying structural lattice on which everything else in the system coheres.

  - Only things that can cohere with that lattice, or with the emergent aggregations on top of it (that is, that coheres indirectly) can stick.

  - Everything else bounces off, free energy not precipitated out of the solution.

  - A new law of physics will attract and stick different bits of free energy into structures unlike the other lattice.

  - It will draw on inputs and energy that didn’t “stick” in the previous lattice.

    - That free energy that can’t be put to good use in the old system will be invisible to people in the old system.

  - To people who are only familiar with the old lattice, the new one will look like it’s creating energy out of nothing.

- It's easier to precipitate a new kind of software on a new structural lattice than to retrofit onto the existing lattice of the same origin paradigm.

- Why do we stick with things that work, and accrete around them?

  - If it ain't broke, don't fix it.

  - Viability means good enough.

  - Viability is more important in practice than quality.

  - Quality only matters if you have viability.

  - If you have a known viable answer and quality doesn't matter that much, why stick your neck out?

- Some conversation partners ratchet up the intellectual level of a conversation, inspiring deeper insights in their conversation partner.

  - Some conversation partners bring the conversation to a more bland, dull place.

  - Does a given conversation partner tend to make conversations they're added to more or less interesting?

    - More interesting means more distinctive, more engaging, further from the bland, average, centroid conversation about something like the weather.

  - If the stakes are high (you have something to prove) a conversation partner that makes a conversation more interesting can be stressful.

  - If the stakes aren’t high it can be euphoric.

  - If you have nothing to prove in a conversation, having an interesting conversation partner can feel like laughing, floating, flying.

- How does your system respond to disconfirming evidence?

  - Does it break, or does it get stronger?

  - That's how you know if it's alive or not.

- It can be tempting to think that because luck played a major role, someone who was massively successful in their first outing will revert to the mean on their next.

  - They likely will revert to the mean, but maybe not as much as you think.

  - The different iterations are not fully independent.

  - Success in the earlier iteration gives resources (capital, knowhow, network) that give a much stronger starting hand in later iterations.

  - It’s still luck but it’s luck with a different probability distribution based on that stronger hand.

  - This can also lead to the reverse misinterpretation: “It can’t be luck, because he was successful in multiple ventures!”

  - Luck, skill, and resources all play a role, and resources can be applied across games.

- Being very powerful or rich is like having an invisible mech suit.

  - They see a problem that everyone else is unable to solve.

    - “I would simply stomp on it and smash the problem!”

    - And their simplistic suggestion works!

    - As the powerful person, you’re dumbfounded why everyone else makes this so overly complicated.

    - “Why don’t you simply do what I do?”

  - They think everyone else around them isn't as smart as them.

  - What's really happening is others aren't as *strong* as them, so moves that are viable for you are not viable for them, and would knock them out of the game.

    - Strength is partially due to the amount of resources they can command.

  - "Simply be bold like me" is easy for the most powerful person to say.

- If your hand you've been dealt or accumulated is super strong, you don't need to be that smart.

  - Your plays will tend to be good even if you aren't being that clever.

  - "Wow, they played that well!" / "Yes, they did, but also it was easy to play it well given how strong their hand is."

- How alive you are is how many kinds of surprises you can successfully meet and navigate.

  - Does your adjacent possible extend to cover that region of surprises you are likely to encounter?

  - If a surprise shows up that is outside your adjacent possible, you have no viable responses, and you either get injured or knocked out of the game.

- If you've made good building blocks then you can respond to surprises and use them to tackle things you never imagined.

  - You're better able to respond to disconfirming evidence and surprises.

    - You have a lot of useful components on hand to combine.

    - The likelihood you have a viable combination in the set goes up combinatorially.

  - Being composed of remixable building blocks, that are viable on their own but have combinatorial possibility, makes you more alive.

  - Building blocks give you a combinatorial coverage in an adjacent possible.

- Human organizations today are layers and layers of human smoke tests.

  - Many of them are redundant, but any one of them could raise their hand and say "hey, this doesn't look right."

  - Similar to the chain mail of prediction loops.

    - It's expensive to have them constantly spinning, possibly even wasteful.

    - But the spinning is what keeps them in a critical state, able to respond and adapt.

  - The ability to respond requires being in a critical state, and a critical state is active, often expensive.

  - Adaptability is infinitely valuable; it's what makes the thing alive, able to grow and become stronger from disconfirming evidence instead of worn down by it.

- I was struck by this example from [<u>Practical Engineering of a bridge that collapsed</u>](https://www.youtube.com/watch?v=4mn0mC0cbi8).

  - The bridge had been reviewed many times by many agencies and problems had been identified.

  - The bridge had *obvious* structural deficiencies: one of the key supports had rusted through and *no longer connected at all to the foundation*.

  - Still, it wasn’t until the bridge collapsed when a bus (with multiple cameras, capturing dramatic footage) drove over it.

  - How did this happen? The bureaucracy was aware of the deficiencies and information was flowing through the process.

  - A bureaucracy is a machine that takes alive responsibility and turns it into dead process.

  - If there is a gap in the machine’s process, a wire that is crossed or where an alert signal doesn’t make it, nothing alive will run into it.

    - “I just ran the report, and am graded on how many I produce a quarter, I’m not empowered to ensure the dire recommendations are followed”

  - A living structure and a dead one look totally similar but work for totally different reasons.

    - The bureaucracy process morphs a living, possibly unfair process to a dead, fair one.

  - The process is highly structured and humans are disempowered in it.

    - No one can say "guys, the steel has rusted and no longer connects to the concrete footing, obviously shut it down"

  - What Dan Davies would call an accountability sink.

- Formally structured systems are good at detecting easy / obvious cases.

  - The ones that get through are by definition the more subtle or rare cases.

    - For example a phone tree often isn’t helpful because it tells you info you can find online.

    - But assuming you are capable of using the web site and did before calling, what you’re usually calling about is exceptions to the rules.

  - The exceptions in a system are the things that matter the most.

  - LLMs are great at looking at in-sample things. But things that are out of sample are the exception!

- In computer graphics, there's always the bottleneck of rendering.

  - As graphics have gotten faster to render, the wait time for rendering stayed mostly the same, because the scope of the graphical effects the artists tackle has gone up.

  - A few related phenomena.

  - Equilibrium of misery: add another lane to a highway, instead of people's commute being shorter as you’d expect, people adjust to live further away (they can get the same amount of house for less money, and they normalize back to the same commute, a quantity of time they know is viable).

  - Parkinson's law: the amount of things you have to do will expand to fill the available space.

  - [<u>Jevons paradox</u>](https://en.wikipedia.org/wiki/Jevons_paradox): “technological progress increases the efficiency with which a resource is used (reducing the amount necessary for any one use), but the falling cost of use induces increases in demand enough that resource use is increased, rather than reduced”

- If you design software by consultants, they'll build fractal individual features.

  - The easiest way to check a box is with an individual thing crafted precisely to address that niche.

  - Doing a factored, artisanal thing that cleanly addresses more than one use case at a time, that expands into unforeseen adjacencies, takes significantly more craft and effort.

  - But it can never create more value than was asked for.

  - A PM with good craft can add value for users in a way the user didn't even know to ask for, might not even know how to describe.

- LLMs impact the “open but illegible” tactic.

  - The open but illegible tactic is to do everything in the open, but in a way that requires the audience to do unpacking and work to understand.

    - This allows a motivated set of users to self-select, without inadvertently setting unrealistic expectations too early.

    - This can help skim the most motivated users into an ecosystem early, giving a high-quality, motivated core to start, and possibly discovering PMF earlier than you thought.

    - A self-capping downside play.

  - This tactic partially relies on the amount of effort to unpack the payload being high.

  - But LLMs are very good at sifting through large amounts of information, and reading between the lines and doing summary and synthesis.

  - That makes the work to extract the throughline of the open but illegible thing lower, which makes it less effective as a check for motivation in the reader.

- A situation: a subcomponent of a larger plan doesn’t fit the expected measurement.

  - Which do you do?

    - Update the larger plan to fit the new measurement.

    - Modify the way you interpret the measurement to fit within the plan.

  - Obviously the former is the correct answer in a fundamental sense.

  - But what if the plan is very very large and has tons of momentum, and if the plan had to change it would require massive amounts of re-coordination work to update the plan?

  - In that case, “fudging” the measurement a bit doesn’t seem quite so bad.

    - Maybe the measurement was wrong anyway!

    - Especially if you, the owner of the subcomponent, will have more to answer for if your measurement is wrong.

    - It’s easier to “go with the flow”.

  - This logic is partly where greenshifting comes from, and is one of the reasons that kayfabe arises in large organizations.

  - A plan that each subcomponent realizes is invalid but which reality (or our measurement of it) has to conform to.

- An inverted webapp architecture:

  - The webapp is statically hosted from a given domain.

  - A user can choose to plug in backend servers of their choosing into the webapp.

  - The webapp stores locally which backend servers the user has configured.

  - The primary domain doesn’t do anything special or have any special permissions.

    - The domain maintains no state on the server, just statically serving up unchanging assets.

  - This pattern feels foreign in the same-origin paradigm, but it’s a totally legal, just a bit atypical, pattern.

  - This pattern is great for things where different users might trust different commodity backends, or where there might be an ecosystem of different providers to plug in, each with a different catalog of content.

- [<u>Gruen Transfer</u>](https://en.wikipedia.org/wiki/Gruen_transfer):

  - "In shopping mall design, the Gruen transfer (also known as the Gruen effect) is the moment when consumers enter a shopping mall or store and, surrounded by an intentionally confusing layout, lose track of their original intentions, making them more susceptible to making impulse buys. "

- A common pattern aggregators do: the peek and poach.

  - Have a sanitized, clean, organized 1P interaction surface area. Then watch what's happening in the bottom-up ecosystem and find the patterns that the swarm has cohered around as good ideas, and then hoist up just those patterns in sanitized versions in the 1P thing.

  - The aggregator lets the 3P swarm do all the work and take all the risk, and then they skim just the cream on top.

    - This then takes all the oxygen away from the 3P system because it overshadows it.

    - As the 3P ecosystem dies out from starvation, the usage of the 1P version increases.

  - This pattern is one that aggregators, given their privileged position, do automatically, without even thinking.

    - Accidentally parasitic.

  - It will be easy for the current crop of proto-aggregators to do this with chat UIs.

  - \#things-aggregators-do