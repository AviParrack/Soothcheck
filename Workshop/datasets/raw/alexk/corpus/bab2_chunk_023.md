# 1/13/25

- *The next Bits and Bobs will be on Tuesday 1/21 due to the US holiday.*

- LLMs burp up middling slop on their own.

  - It is up to the human co-creating with them to drive them to something interesting and curated and good.

  - LLMs have no filter for what is good.

  - You need humans in the loop to apply taste and select the good stuff.

  - The quality of output from an LLM is highly correlated with the quality of the human using them.

- Talking to an LLM feels not like talking to a person, but talking to the collective hive mind of humanity.

  - LLMs give great guacamole recipes, because they’ve seen *every* guacamole recipe and can triangulate what they all have in common.

- LLMs shouldn’t help you do less thinking, they should help you do *more* thinking

  - They give you higher leverage.

  - Will that cause you to be satisfied with doing less, or driven to do more?

- LLMs allow you to scale your taste by giving you higher leverage in creation.

- Treat LLMs as a dance partner, not an oracle.

  - Oracles give you a fully formed answer, a one way information flow.

  - A conversation partner you discuss things with together, collaboratively, an intellectual dance.

  - The result can be better than either of you could produce alone.

  - You can have a really strong partner who makes you look better.

  - But you still won’t compare to two really strong partners dancing with each other.

    - One strong partner can make up for deficiencies in the other, but two strong partners interacting is an order of magnitude better than either apart.

  - Intelligence and output becomes a co-evolutionary dance between the user and the LLM.

- You can't mindlessly use an LLM and hope to get anything more than mindless results.

  - The mindset and concepts you bring to the LLM determine whether it gets good results.

  - If you have a coherent worldview and concept, the LLM can help you fill it in.

  - If you don't, the LLM will hallucinate the concept for you.

    - This hallucination of a background worldview is dangerous!

  - The editor's mindset is more important than the writer’s mindset in this process.

- The human and LLM interaction is about co-creating.

  - Someone was telling me they’ve started writing poetry again after 20 years, because it’s so much easier to co-create with an LLM than to do it alone.

  - We’ll see an explosion of amateur poets and musicians co-creating with AI.

    - Making it so it's easier to do, higher leverage, less intimidating..

  - Even if no one reads their poetry, the act of writing poetry improves the world.

  - An amateur poet becomes more observant of their world, they become a better, more engaged, more thoughtful participant in their environment.

- Co-creation between a human and LLM is related to Hegel’s dialectical process.

  - The human and the computer, the distilled hive mind of all of society, in a co-creative process.

  - Fundamentally different, fundamentally better, than a computer talking to a computer, or a human talking to a human.

  - The tension and difference in skillset is what produces novel insights that could not have been created with a pair of similar conversants.

- LLMs make it so you’ll never be lost intellectually.

  - Intellectual GPS.

  - When you have a co-creative muse, you never feel lonely or lost in your creativity.

- AI is great for encouraging curiosity.

  - Any question you ask it. it will reward you with some thought.

  - Kids should get used to “have some curiosity? Go ask the LLM!”

  - A massive complement to “sit here in your seat and do rote memorization to pass the test.”

  - Embracing the joy of discovery.

- People who think LLMs can’t be useful either haven’t tried them or aren’t curious enough.

  - Curious people (especially ones with a need for cognition) will keep pulling on threads and will love them.

- Google-fu is a temporary phenomenon.

  - When Google was a new thing, people could have “Google-Fu” who had developed a calibrated intuition for how to extract better answers from it.

  - But as Google matured, a clear gradient of improvement is to make it so less-and-less savvy users get results closer and closer to what users with Google-fu can get.

  - The main way to do that is to automate the intuition that the savvy users are applying; which now means everyone gets it, and the edge evaporates.

  - Prompt-fu is the equivalent for getting much better results from LLMs.

  - That implies that people with prompt-fu will get less and less differentiated results as the systems we all use get more mature.

  - But presumably there will always be some difference, because a search is just a filtering operation, whereas LLMs are co-creative, so the “Fu” is orders of magnitude more important than it was for search queries.

- Transformers are unreasonably good at extracting patterns.

  - Apparently if you train them on RNA sequences to images of the rendered protein, they do a surprisingly great job at predicting what a given sequence will fold to.

  - LLMs are tapping into a hidden structure of the universe that reveals itself only if you are patient enough to sift through it.

  - LLMs are patient, and observant enough, to tap into that structure.

- How much does background world knowledge affect an LLM’s ability to summarize text?

  - For example, how much does it matter to use a model that implicitly knows an elephant is bigger than a mouse?

  - You can think of summarization as a process to factor out the background knowledge a reasonable listener would take for granted, leaving only the “diff” of interesting meaning.

  - That implies that the more the LLM understands about the world what a typical human does, the better the summarization will be.

  - Presumably that ability has a clear logarithmic shape, where more background knowledge gets less and less useful.

- What is the value of proprietary information included in the training of an LLM?

  - That information helps the LLM perform better, but how much?

    - How much worse would the LLM be if you hadn’t included that marginal bit of data?

    - Would a human even notice?

  - Someone pointed out that the [<u>Shapely Value</u>](https://en.wikipedia.org/wiki/Shapley_value) might be a useful conceptual lens to try to get a handle on this.

- GPT-4 level quality can now safely be considered a commodity.

  - That is, lots of options, good price, and quality competition.

  - That's an amazingly optimistic result for society!

  - We can take GPT-4 level quality for granted.

- There are a lot of new techniques to “frack” LLMs to squeeze more out of them.

  - If you get just a single english-language append-only log designed for a human, you make the LLM take its rich understanding and distill it out through a teeny straw of a single, human-understandable line of thought.

  - Techniques like test-time compute in O1 and similar models allow the model to spray out lots of low quality ideas and then refine.

- You can view LLMs as a lemon to squeeze.

  - The quality of an LLM’s juice is tied to who is doing the squeezing.

  - An implication of all outputs from LLMs being co-created by the human and the LLM is that different humans can get wildly different results.

- A model being trained on data or using RAG at inference time has wildly different characteristics.

  - But a lot of discourse about LLMs doesn’t differentiate the two.

    - I see even technical people muddle this all the time.

  - There's a difference between an LLM in training absorbing a hologram of the knowledge vs RAG to help sift through concrete input with its background common sense it absorbed in training.

  - Sometimes you just need its background worldly knowledge to give it common sense.

  - If you want details, that's not sufficient and you’’l need RAG.

  - Adding more knowledge to a model is expensive, has long lead times, works on vibes and is imprecise.

    - The larger the model, the less that any incremental bit of data in training affects the output.

  - RAG can't give huge context to a model that doesn't have the right background knowledge, but it can be updated quickly and can enable precision in details.

  - Everyone talks about these things like they're the same, but they're wildly different.

  - Training your own model is very capital intensive.

  - But in many cases you can use an off-the-shelf LLM plus RAG and produce amazing results.

  - The question is: how much background knowledge do you need for the LLM to have enough common sense to be able to tackle your concrete tasks where you bring the specific details for it to operate on.

- In some ways AI is naturally centralizing.

  - The centralization is implied due to the capital requirements of training and inference, the existence of proprietary models, and the efficiency of scale in serving.

  - Centralization would be bad if there were stickiness to models.

  - But if you treat the model as a dumb, stateless machine, giving an answer to your prompt and then forgetting, it doesn’t matter much; they don’t accumulate data to accumulate power.

  - It’s once the model starts getting a memory that the power dynamics turn into something possibly compounding.

  - The fact that LLM providers are now commodity and also that the API is the same and easy to swap to a different provider at the flip of a switch helps reduce the likelihood of centralization of power.

- Last week I [<u>compared LLM providers to electricity providers</u>](#bjqy4mufhfcl).

  - Capital intensive, but commodity.

  - But actually LLMs are a worse business than electricity providers.

  - Electricity providers typically have a geographic monopoly; end consumers only have one option to pick from, which gives some pricing power.

  - But LLMs providers don’t have that, it’s super easy for someone to swap providers in a second.

  - It’s even the same basic API (prompt -\> response).

- AI is the app killer.

  - It will cause an app unbundling.

  - It reduces the “transaction cost” of creating software, which, similar to the Coasian theory of the firm, sets the efficient “size” of bundles of software.

- One model of AI unbundling apps is that now you'll have swarms of agents poke at and slurp from apps on your behalf.

  - You won't see the apps much... they'll still be there, just kind of boring basements below the layer that you the user spends time in.

  - But that implies that you have agents who can see and intermix data across apps, and that you trust to not take incorrect actions on your behalf.

  - That’s a high bar to meet if there's even a little bit of downside risk... and if they're flexible and open-ended there's always downside risk!

  - Another approach is not an over-the-top of existing software, but new software emerging.

  - Perfectly bespoke software on demand.

- Imagine: Your data, alive, animated.

  - Sprinkle pixie dust on your data, it comes alive.

  - The software is the least important part.

  - The software is a means to an end, an implementation detail.

  - Something that emerges, that you can take for granted.

- Imagine: magical instagram filters for your data.

  - Instead of making the data look pretty, it makes it animated, interactive, useful.

  - Instagram filters give you just the right amount of agency.

  - You feel like you are making good decisions... but the decision space has been constructed so all of the decisions you could make are good.

  - You are imbuing it with your taste with high-leverage tools.

- Imagine: as you navigate through your data and make lots of small curatorial decisions, you are implicitly constructing software.

  - Even if you don't realize what you're doing, and would never consider it “programming”.

  - Curation and micro-decisions as an act of creation.

  - Supercharged with AI, it allows software to emerge, implicitly called into being just in time.

- We need a new realm for computing.

  - One with different rules, a different gravity.

  - Where things that in today’s world would be considered magic are ordinary.

- As things scale they regress to the mean.

  - If you want to grow your audience, you have to get closer to the lowest common denominator.

  - This is inescapable.

  - An alternate way to express Ivan’s [<u>Tyranny of the Marginal User</u>](https://nothinghuman.substack.com/p/the-tyranny-of-the-marginal-user).

- We need organic software.

  - The way that traditional software is made is fundamentally and intrinsically bad for you.

    - Either trying to trap you or increase your engagement against your will.

    - Giving you what you want, not what you want to want.

    - Engagement, not meaning.

  - Organic software should work just for you, and should be healthy.

- Email is our personal informational compost heap.

  - Email is extremely noisy, but also tons of signal.

    - What you decide to subscribe to, even if you don’t read it.

    - What services you use.

    - I’m not the only person to have [<u>noticed this</u>](https://bsky.app/profile/fancypenguin.party/post/3lfdszzuruc2o):

      - "Email is the only protocol on the Internet that centers the individual. Email clients, however, have not evolved to recognize the ways in which email is used. Your email is a todo list, a library, a recipe book, a transactional history, and more."

  - What if interactive, useful software just for you could emerge out of that compost heap?

    - Like an ecosystem of friendly bacteria, a slime mold.

  - At places like Google it’s impossible to do this because although they have your data, they would have to *build,* not *grow*, software.

    - When you build software you need PMs to figure out a piece of software that will have a large enough market, and engineers to build it.

    - Software in that style requires users to have a stranger with an ulterior motive be able to see their stuff.

    - It also runs into the tyranny of the marginal user.

      - As the provider scales, it gets harder to coordinate on planning small things, and everything regresses to the mean.

    - What if instead, software could grow?

- “Yes, and” in practice is often “yes, and… (my idea)!”

  - The true "yes, and" is “no matter what you say, I’m going to embrace it and make you look good”.

  - “Yes, and” should be building on top of, co-creating with the other, a thing that is better than what either of you could do alone.

- You get to set the rules of your game.

  - But others get to decide if they want to play in your game or not.

  - This naturally balances out incentives.

  - You have to do something that others will want to participate in.

  - Like the optimal process for fairly splitting a dessert that all kids know intuitively: one kid splits, the other kid picks.

- In low-friction environments, higher quality things have compoundingly better outcomes.

  - The seed crystal exists because the first few people who looked at it found it useful.

  - The boundary gradient works because people at the margin can see that the thing is useful and is not hollow.

    - They use it because others have found value in it and it’s useful to use the same thing others already like… but also because when they look at it they can see it seems useful to them, too.

  - The quality of the thing thus does correlate with how successful it becomes, but the quality to outcome is not linearly proportional (due to the compounding being a self-accelerating phenomena) but a power law.

  - That's why power laws happen; the boundary gradient's steepness is driven by quality (a linear difference) but the size of growth is driven by surface area of the boundary, which grows with the square of the current size.

- Regulations cap downside. Benchmarks set the terms of how to measure upside and inspire competition by making it measurable.

  - Benchmarks are an emergent schelling point.

  - Someone sets rules and a way to measure quality.

  - No one has to use their rules if they don't find them valuable, but if people do, then other people will also want to show they can do well on it, which is a compounding loop.

  - People take it seriously because other people take it seriously, and people take it seriously because every marginal person who considers taking it seriously looks at it and agrees that it sounds plausibly useful enough to take seriously.

  - Benchmarks can get a compounding amount of momentum in proportion to their quality.

- Digital commons get stronger with more use, vs physical commons, which get worse.

  - Because data is non rivalrous, it doesn't suffer from the tragedy of the commons.

  - Only atoms have the tragedy of the commons, not bits.

  - Bits are non-rivalrous in their consumption; you can make infinite perfect copies for free.

  - So internet bandwidth can have the tragedy of commons, but not the information carried over the wires.

  - Digital commons tend to get more investment from an ecosystem the more use they get.

  - The more that others use them, the more that people are invested in making sure information they care about (e.g. information on their hometown) is high quality.

  - People invest in proportion to how useful other people find the resource to be and how much traffic it gets.

  - So the more people use it, the more value is naturally created.

- For a fast-moving technical system, safety sometimes gets left by the wayside.

  - You could argue that because there are benchmarks for e.g. LLM safety, LLM providers will also compete on doing well on those, naturally.

  - Safety is a thing that people want to want, but don’t actually want.

  - If people get an incremental personal benefit from using an unsafe system vs the competitor who uses the safe system, they’ll take it.

  - That leads to an arms race on capability leaving behind “safety.”

  - "Well if I, the good guy, don't push the limit, the bad guy definitely will and then dominate us all, so I need to push the limit so the good guys win."

    - That leads everyone to push the limit.

    - *Everyone* thinks they’re the good guys.

    - Except that one guy in that meme, who realizes he might be the baddie.

- Competition serves as a natural regulation.

  - The co-evolving set of competitors hold each other in check.

  - Competition sets co-evolving constraints that are balanced and just right to encourage a gradient of improvement that pushes all competitors to excel.

  - But it's a bowl on a pedestal kind of dynamic; within the middle range the competition is self-balancing, but if you get out of the place where it’s possible for competitors to catch up, the engine that leads to quality growth leads instead to run-away power accretion.

  - By using the power-accretion gradient and aligning with quality improvement, everyone improves due to individuals’ greed.

  - But greed without the constraints from effective competition consumes everything.

  - It is only the competition, the dynamic equilibrium, that drives to quality creation, the positive-sumness of greed.

  - Without competition, greed just consumes all and puts everything into a fully captured heat death kind of static equilibrium.

- Financialization creates efficiency, but is in tension with potential greatness.

  - It’s better able to invest in things that are working; but it can get much harder to iterate and experiment and see what's working.

    - "Before we let you do this prototype, what is the ROI of this in 5 years?"

  - Financialization can make it very hard to get the seedlings to be able to garden in the first place.

  - Financialization is about hyper legibility, hyper focus on quantitative.

  - But sometimes you don't need it, because it's more expensive to create legibility than it is to just do the thing.

  - In the time it would take to document a seedling, you can plant 10 seedlings.

  - As long as the downside risk of the seedling is capped and small, it's just opportunity cost, and making it fully legible just kills the seedling before it even gets started.

- Capitalism (which is just an outgrowth of evolution and the Technium) is best at delivering the “most.”

  - The most of just whatever our lizard brains want.

- Political scandals of the past seem so quaint.

  - Being chased by a rabbit. A kind of weird yell at a rally. Misspelling potato.

  - Why are they so much more tame?

  - I think it’s because of the internet.

  - The information you are exposed to daily sets your baseline for normal, for what is exceptional or stands out from that baseline.

  - Before the internet, most information went through curatorial processes optimizing to make them normal / balanced.

    - If there are only 3 broadcast TV channels, they all have to cover most of the bell curve of the population; a strong pull towards the center.

  - The internet makes it so there’s infinite channels for information to flow, and also a constant drive to produce things that are “more”.

    - You get things like the Doritos Crunchwrap Supreme of information.

  - The things that break through the cacophony are memeable, the “most”.

  - That means we’re awash in a cacophony of over-the-top, supernormal stimuli and that becomes our new normal.

  - Now that the baseline is over-the-top, it’s the shameless who have adapted to the new reality the fastest.

  - The shameless have realized that even egregious behavior barely registers, so the downside is much smaller, and it’s a much better strategy than before.

  - Before, doing the right, principled thing, and the thing that was most effective in repeated games were aligned.

  - Now, they are less aligned, and the shameless reign supreme.

  - I truly hope we as society figure a mechanism to realign principles even in this new information reality.

- Democracy only works if people believe in it and execute it.

  - The rules don’t follow themselves or apply themselves.

  - They require people who believe in the rules to hold themselves to them or apply them to others.

- Leaderless emergent movements almost always overstep.

  - There’s no regulator to slow down the process, especially if there’s a moral fervor about righting some persistent wrong.

  - Everyone is swept up in the momentum, even as they increasingly lose conviction that it’s still justified to the extent the movement is executing on it.

  - Going against the momentum, against the stampede, now becomes dangerous (you’ll get trampled) so you go along.

  - Only after the mob has clearly and ambiguously crossed the line can everyone go, “oh, yeah… that was obviously too far.”

  - Unfortunately that tends to invite radical over-reactions from the other side.

- Schelling points are lighting rods for luck.

  - The thing that is most prominent, most distinctive, wins by default in the chaos.

  - I was talking to a person at CES who had an amazingly lucky break: a job offer for a cool and extremely differentiated role.

  - She got it when she was the only American living in a particular region of China.

  - She stood out, which meant she was structurally more likely to get noticed.

- Noise matters less than consistency.

  - When you have a consistent, asymmetric edge in a dataset and average it across many, many, many iterations, the edge pops out, clear as day.

  - The more data, the more convergence and revealing the true, consistent edge, no matter how much noise.

- A prototype is kindling.

  - It’s not the end thing, it’s the catalyst to propel you to the main thing.

  - It's combustible, you couldn't get it started without it.

  - But it is consumed in the process, to create the fire to get the actual logs going.

  - The prototype is ethereal, self-immolating.

  - It's just about consuming itself to create momentum.

  - Don’t focus on making it perfect; focus on making something combustible.

- A possible iron triangle: safe, general purpose, and people want to use it.

  - People wanting to use it is the most important one.

  - You can’t force people to want to use it; it has to emerge, organically, if you’ve built something that is viable and useful.

- Don't focus on the ambiguity, focus on the concreteness.

  - Lock the parts down you can, getting a larger and larger base of concreteness.

  - That foundation will allow you to reach further into ambiguity.

  - You need something concrete to lever off of to stay strong in ambiguity.

  - A mistake a lot of people make is to focus on the ambiguity instead of what can be easily nailed down.

  - Ambiguity has compounding cost, so nailing down more pieces makes remaining decisions orders of magnitude cheaper.

  - Nail down the most obviously true, inescapable parts first, and work your way down to ever-more debatable points.

  - As you work your way down and get more real world feedback, it will get easier and easier per unit ambiguity.

  - Don’t postpone decisions you don’t need to.

- In ambiguity, when you feel strong, you get more flexible.

  - In ambiguity, when you feel weak, you get more tight, defensive.

  - The defensiveness creates brittleness and makes you less resilient.

  - You need to feel strong to act strong.

  - A kind of inherent chicken and egg that can lead to a vicious or virtuous cycle depending on your natural disposition: how optimistic you are.

- If you present yourself to others as having all the answers, you've trapped yourself.

  - You can't say "I don't know."

  - So you need to go to increasingly ridiculous, unproductive lengths to avoid saying “I don’t know.”.

    - Dysfunctional levels of detail.

    - Avoidance behaviors.

    - Passive aggressiveness.

    - A defensive crouch.

  - Even worse than needing to have the answers is needing to be the one who comes up with the answers.

    - An absolutely impossible bar to clear.

  - Being able to embrace that you don't have all of the answers is a strong stance that requires self-confidence.

    - Without self-confidence, you can't be strong enough to say it, to set yourself up to succeed and grow.

  - Acknowledging “I don’t got this” takes a lot of self confidence and trust.

  - This is one of the reasons the two unteachable skills need an optimistic, forward momentum: to get that chicken or egg self-confidence and strength coevolving earlier.

- Practice makes it *easy*.

  - When it's easy, when it's not a chore to push beyond your current ability, to apply discretionary effort, to go beyond.

  - When it’s effortless to excel is when greatness can emerge.

- A second-hand summary of parts of Rick Rubin in *The Creative Act:*

  - When you don’t have the muse don’t worry about it or try to get it to show up.

  - When it strikes, push it as hard as you can and be thankful for it.

  - Don’t stress when it’s gone, it will come back, and you can't force it to come back.

  - All you can do is make the most of it when it does hit.

- Simple and easy are disjoint.

  - Simple is how hard it is to understand the right answer.

  - Easy is how hard it is to execute the right answer.

  - It's totally possible for something to be simple but not easy: a slog, or an unforgiving domain where the action is easy but most attempts end in failure and the only way to get good is to develop the knowhow and feel.

- Imagine watching someone do a bunch of random, kind of sloppy actions.

  - Then, unexpectedly, magic happens.

  - In a flash you realize: this person was not flailing, but doing intentional, precisely controlled steps the whole time.

  - A perfectly executed magic trick.

  - Every seemingly random movement was dialed in to call the magical result into being.

  - You have an ‘aha’ moment, as the bisociation collapses.

  - You go from thinking “this person is just making it up as they go along, with a lot of noise” to “this person is in very tight control, and I should pay attention to every detail.”

  - That aha moment can be very compelling and make you trust someone deeply.

- When little kids go to school for the first time, their language gets a step change in clarity.

  - When kids are learning to talk and they’re just talking to their parents their parents understand them even when they’re unintelligible to others because the parents remember what that sound signifies from direct prior experience.

  - The parents are reacting less to what the utterance sounds like and more recognizing it from what they know it's connected to.

  - If you say a thing and it's unintelligible to the receiver, you have to modulate and try again.

  - If the receiver understands it, there's no need to modulate, because it achieved its goal.

  - But being with a bunch of peers, if they can’t understand you you need to change how you speak.

  - It’s only by failing to be understood that the child has an incentive to produce the utterance more clearly.

  - English is an equilibrium of being mutually intelligible.

    - Everyone knows what you mean with the utterance because it has to be mutually intelligible to people who haven't heard you in particular speak before.

  - So kids at school have much better language production.

  - They have to be intelligible to a lot of other people who don't have the time or mental capacity to remember every random person's specific speaking style.

- Apparently today 60% of energy used in computation is used to move memory into and out of where the logic is actually executed.

- For large organizations to operate effectively, they need some slack.

  - The slack helps absorb surprise without a compounding cascade of thrash.

  - It also allows little seedlings of greatness to grow.

  - Don't think of slack in a system as "employees sitting around twiddling their thumbs” or “doing unimportant hobby projects” or “playing foosball".

  - Think of it instead as "doing preemptible work that is important but not urgent."

  - Doing the P2s. Which, over time, can add up significantly.

- A tension: efficiency vs thoroughness.

  - In a small company, you can be efficient but you can’t–and shouldn’t–be overly thorough.

  - In a large company, you can’t be efficient, and you should be thorough.

    - You have more downside often so thoroughness is more important.

  - The worst of both worlds: inefficient and not thorough!

- Don't try to think too much about what has happened to get us here.

  - That can easily become navel-gazing and apportioning blame.

  - Focus on how to get the most out of where we are, given that we're here.

- How many days a week do you have to spend on organizational BS to get great execution?

  - At 1000+ it's 5 days a week.

  - At 100 it should be 1 day a week.

  - Up to 40 people it should be zero.

  - If you’re under 40 and it’s more than zero, something is wrong.

  - Politics emerges in every assemblage of people, but it should not become a dominant force until you’ve grown to a size where not everyone can have a direct, personal trusting relationship with each other person.

- When you're with people who are comfortable in their own skin, it's so fun.

  - Even if the people are very different, they can have fun together.

  - People who are defensive or trying to be something they aren't, it's not fun, it's like swimming upstream.

- To help people thrive they need structure and support.

  - Not a cage, a scaffold.

  - Some environments already have some structure you can take for granted, and all you need to do is add fertilizer.

    - For example, post-coherence teams, larger organizations, pre-existing teams.

    - Things that would exist even if you personally stopped working on it.

  - But in early stage, pre-coherence contexts, you can't take any structure for granted.

    - You have to create it.

- Innovation requires exploration.

- One thing I take for granted working in software: very short prototyping times.

  - In the world of bits, you can go from a rough idea to a rough prototype often in just a few hours.

  - This makes it much easier to explore, find compelling ideas, and experiment.

  - In the world of atoms (e.g. hardware), it can take many, many, many orders of magnitude longer between having the idea to determining whether it’s actually promising.

  - The longer between the decision to invest and determining whether it worked, the more like roulette it is.

- Do the plumbing before the poetry.

  - Survive, then thrive.

- No matter how good you are and how hard you work, if you're in the wrong context then it's just not possible to do your highest and best use.

  - When you’re in your highest and best use, you’re in your flow state.

    - What’s good for you and what’s good for your context are deeply, sublimely, aligned.

  - Sometimes the context is a thing you can control, but often it isn't; you need the support of others to help make that context correctly aligned.

  - Sometimes the context that two high-ability people need to thrive are simply incompatible.

- A friend of mine: “I'm less of a systems thinker, more of a systems *feeler*.”

- What do kids mean when they say "That's not fair"?

  - “I don’t like it!”

- Quoting in full a [<u>response from Jake Dahn</u>](https://docs.google.com/document/d/1GrEFrdF_IzRVXbGH1lG0aQMlvsB71XihPPqQN-ONTuo/edit?disco=AAABbOFizyY) about last week’s riff on tension and abrasion:

  - "abrasion becomes tension if the other person isn't ready for it

  - nobody wants a scraped knee / ego, so more often than not it ends up being tension anyway

  - trust is the antidote"

- When you are feeling a feeling, name it so it doesn’t control you.

  - Point at it, hold it.

  - Acknowledging it allows you to hold it at arm’s length and move it to a has-a relation.

    - “I am feeling angry”

  - If you don’t, then it is implicitly part of you (is-a) and can have strong implications for how you act that are often not what you want to want.

    - “Jeff is a jerk and I want to sabotage him to get even.”

  - Don't ignore emotions. Name them, label them, hold them as an object.

- There are no easy ways to learn.

  - Failure is the only way to learn.

  - Failure hurts.

    - Disconfirming evidence.

    - Confronting that you were wrong.

  - Don’t try to make no mistakes.

  - Try to fail at the fastest positive rate to learn.

  - Fail early and often.

  - Smaller mistakes that are less likely to knock you out of the game, a faster feedback loop to learn.

- Vibes that really resonate with me: this comic about “[<u>The obstacle is the path</u>](https://x.com/raminnazer/status/1876722408813474053).”