# 10/14/24

- When wizards prompt an LLM, it’s like a magic trick.

  - People watch the results and they look like magic.

  - But then you show them the prompt you used to do it, and the onlooker goes, “... oh, that’s it?” and feel almost disappointed.

  - But unlike a magician, there is real magic going on.

  - It’s just that the LLMs are what is magic.

  - The wizard just knows how to marshall that magic effectively.

  - If you want to impress people with your LLM wizardry, don’t show them the prompt!

- The things that lead to the transformative consumer usage often are not about tech but distribution.

  - They take an existing bit of tech that is only available to very savvy users and make it available to anyone.

  - The distribution innovation will look almost like a non-accomplishment to the savvy users.

  - “You didn’t make anything possible that wasn’t possible before!!”

  - But you did better: you made a thing that was previously only possible, *easy.*

  - Now a much larger set of people can use it.

  - LLMs are amazingly impressive, but to use them effectively you have to be a wizard.

  - The tool that has the biggest impact will allow *anyone* to do what an LLM wizard can do.

- LLMs sweet spot: a task that any reasonably competent and worldly person could do without a lot of mental effort, just a lot of *time*.

  - For humans it would require a lot of System 2 effort to do the research to find the facts.

    - Which would take tons of time and analysis, even if each bit of research was superficial and straightforward.

    - There’s few things that clear that bar to be worth it.

  - But LLMs are a society-scale System 1.

  - They can do associative reasoning with no effort at all.

  - Suddenly all kinds of analysis that were impossible before become easy.

- Qualitative and quantitative analyses aren’t *that* different, fundamentally.

  - Quantitative emphasizes getting measurable results.

  - Qualitative emphasizes getting deep results.

  - In practice it’s expensive to get both, so which methodology you emphasize leads to what results you get.

    - Quantitative: measurable but shallow, surface level summary statistics.

    - Qualitative: deep but narrow.

  - But now LLMs can do mediocre (but robustly mediocre) human-ish analysis of non-numerical data.

  - So you can get qualitative style depth with quantitative level measurability.

  - When the fundamental cost structures of things change, it messes with our intuition of what’s possible.

  - All of a sudden things we just implicitly assumed were fundamentally impossible become, surprisingly, possible.

  - It will look like magic.

- Terence Tao says that he has a curiosity that allows him to go deep in lots of areas, to draw connections that other experts can't find.

  - A generalist can connect dots that weren't connected before not because the dot is hard to connect, but because no one thought to try before.

  - The combinatorial possibility space across disciplines is impossibly large.

  - As humanity we’ve only tried a small subset, because human effort is expensive.

  - But LLMs can do a mediocre human analysis, cheaply.

  - So we can find the low-hanging fruit that was pre-existing, we just didn't find yet.

  - Even if LLMs are just able to draw connections between different disciplines, that could be useful.

- A thing I want to ask LLMs to do (in a series of prompts):

  - 1\) enumerate 100's of academic disciplines

  - 2\) for each list the 20 big ideas (replicated, core, differentiated ideas).

  - 3\) For each combination of insights from across domains, ask the LLM, "draw parallels and discuss any interesting overlap, and suggest further areas of insight this combined idea might imply”.

  - Most combinations will be meaningless or uninteresting, but some will be game changing.

  - So far we’ve used humans swarming on large scale problems to find and try the interesting combinations.

  - But humans are expensive and also take time to come up to speed in a different discipline’s jargon and knowledge, so we’ve tried only a tiny subset of combinations.

  - But LLMs have read all of the disciplines and also are cheap.

  - So as society we can now structurally find significantly more interesting and game-changing combinations.

- You know many experts in your network who are the absolute best at a given topic.

  - A good way to approach those problems is to ask yourself “What would X do in this situation?”

  - For example, for engineering architectural issues, I ask myself, “What would Dimitri do?”

  - You can get better and better at this over time.

    - For each individual situation, form a hypothesis of what Dimitri would say.

    - Then ask Dimitri.

    - If you got it right, great! If that happens a lot over time, you’ve successfully absorbed a simulation of Dimitri’s knowhow.

    - If you got it wrong, update your model.

  - This process is finicky and takes a lot of time.

  - But imagine that this expert has written quite a lot about their area of expertise.

    - This is rare, [<u>but it happens</u>](https://whatdimitrilearned.substack.com/)!

  - If you somehow had read all of their writing and could recall all of it, your prediction would be way better.

  - That’s hard for humans to do… but easy for LLMs!

    - Just feed it all of the embeddings of their writing, and it can do a convincing facsimile of that expert’s reasoning.

  - If you had that, you could assemble little virtual rooms of your preferred experts on different topics and have them discuss the answer.

    - A “Gordon” to discuss product and design issues, a “Dimitri” for the architectural issues, an “Erika” for the emergent org and cultural issues…

- The information flow rate in a conversation is tied to the human brain's ability to produce and consume speech.

  - You can think of this as the information per unit time, the clock speed.

  - Some people are incrementally better or worse at it, but all humans are in the same basic ballpark.

    - I asked o1 to estimate the number of bits of entropy per second in typical human speech, and it [<u>suggested 33 bits</u>](https://chatgpt.com/share/6707e55e-6080-800f-a33e-cd1788fa1ff8).

  - Embeddings are a denser encoding of richer meaning, and computers can have a faster clock speed to unpack it.

  - This would allow denser communications.

- There are different pace layers for getting results out of LLMs.

  - Slowest: train a new foundation model from scratch.

    - Also extremely capital intensive!

  - Medium: fine-tune an existing model

  - Fast: Prompt engineering

    - Carefully calibrating the exact structured guidance you give to the LLMs to get it to give reliably good results for a class of problems.

    - You can do patterns like few-show learning.

  - Fastest: Context engineering

    - Precisely what extra background knowledge you pass it to help it answer *this* question.

  - People talk about fine-tuning and making new foundation models, but the last two layers are remarkably effective, and wildly cheaper to accomplish.

  - Now that context windows are so large, you can get very, very far with careful prompt engineering, few shot examples are extremely effective.

    - There’s tons of space to give the models carefully crafted examples!

  - Don’t assume you have to dip to lower levels before you actually have to!

- The gold standard is a system that improves its quality the more it's used.

  - Self-ratcheting. Antifragile.

  - The system has to be open-ended. Able to absorb disconfirming evidence and new ideas to improve itself.

  - LLMs, as they exist today, cannot.

    - They are closed systems, sealed off at the time of training.

  - But you can still use the LLM's magic and *wrap* it in an open system that can self-ratchet in quality with use.

  - Another bit of evidence that the LLM is not The Thing, it's the *enabler* for The Thing.

  - LLMs are the missing magical ingredient that makes the new Thing possible.

- The mark of true quality is the closer you look, the more compelling it becomes.

  - Things with only a veneer of quality will become less impressive the closer you look.

  - They are hollow.

  - In the worst case, gilded turds.

- Imagine there’s a high taste and innovative playbook someone has.

  - Their own particular “taste,” reified into a playbook.

  - It produces output that is wildly outside of the status quo.

  - You see one result from the playbook and it’s super compelling, sui generis, innovative.

  - But now imagine seeing 20 examples back to back.

  - You’ll see the pattern, the structure, more than the individual examples.

  - No matter how innovative and high taste the playbook is versus the general status quo, each example will feel similar to each other, making them feel more bland, limp, lifeless.

- I find NotebookLM fascinating.

  - The closer you look at the results, the more you see the patterns, the more you realize it’s an uncreative process that is a cocktail of existing good podcasting tropes.

  - The closer you look at the outputs, the more you realize they are mediocre… but *reliably* mediocre, wrapped in a charismatic, superficially high-quality package.

    - Not unlike MidJourney results.

  - Masterfully applying useful tropes producing something that has superficial trappings of quality.

    - A checkbox can't produce upside, it can only avoid downside.

  - NotebookLM can produce a superficially high quality, engaging, charismatic podcast out of *anything*.

    - But it doesn’t necessarily have anything interesting to say about anything.

    - [<u>Someone fed it a long file</u>](https://futurism.com/the-byte/google-ai-podcast-poop) that just said “Poop” and “Fart” and it still generated compelling output.

    - That’s how you know it’s able to put a charismatic and engaging podcast-formatted wrapper around anything, without necessarily having anything to say.

    - An engaging dialogue between two open, worldly people is just inherently compelling to our lizard brains.

  - NotebookLM is clearly constructed by people who have a highly calibrated taste for compelling podcast content.

    - That gives a veneer of taste to every bit of output.

    - But a mechanistic, unchanging object can’t produce output of true taste on its own.

    - True taste is sui generis; it adapts, it evolves.

    - NotebookLM is presumably a bundle of carefully tuned prompts and a few generic underlying foundation models.

    - Those mechanistic guts aren’t alive, no matter how superficially captivating the results are.

- LLMs are trained on the past.

  - They can reproduce things that were novel and high taste from the past, but not the current frontier.

  - Taste constantly evolves, to allow people with taste to differentiate themselves from the people going with the new status quo.

  - Everyone is constantly seeking an edge; some people’s variance turns out to be good, and then others follow along, and over time it becomes the new status quo.

  - If you bring everyone up to the average, you need someone with the discernment to see the edge.

- The LLM is a hyper-dimensional possibility mirror.

  - The average of all of society.

  - To squeeze great things out of it, you have to know how to drive it.

  - The average driver of an LLM will get mushy, average results.

  - But the best drivers of it, with calibrated taste and knowhow from experience working with it, will get magical, great results.

- A child prodigy is in novice mode, they have the benefit of not knowing what's been explored, their takes are uniquely novel.

  - But LLMs can never have that advantage, they can instantly see everything that came before in their training.

  - But if you drive them into useful things via external structure (e.g. particle colliding) or taste, they can find novel combinations.

  - A prodigy is all upside: if they're bad, people say, well they're a kid.

  - If they're good, people say, "wow, a *kid* did that?"

  - A prodigy is about not realizing what the quality of the thing is, a childish naïveté.

  - Most breakthroughs are people didn't realize were generally thought to be impossible.

  - How can you make the LLM act like a child prodigy?

  - “Wow, an *LLM* did that?”

- Vibes and taste are distinct.

  - LLMs are great at vibes.

    - They’re computers for vibes.

  - But LLMs are terrible at taste.

  - LLMs are high on vibes, low on taste.

  - Taste requires you to be good at vibes, but vibes are not sufficient.

  - Taste is mastery of your own personal vibes.

  - But if your personal vibes are "everyone's thoughts ever" then it's just the boring centroid.

- You can't develop your own taste by just turning the crank real fast.

  - Execution and creativity are very different.

  - Satisficing / playing not to lose vs playing to win.

- The zone of proximal development is magic.

  - Too far beyond your ability: impossible, frustrating, noise.

  - Too close to your ability: boring, safe, uninteresting.

  - Proximal development: interesting, challenging, growth.

  - In humans the zone of proximal development is called the “flow state”--a resonant emotional experience, you “feel like a million bucks”.

  - But other systems have it as well, and it’s magical for them, too.

  - The zone of proximal development is the goldilocks zone of maximum growth and learning.

    - Easy enough to keep at it; hard enough to stretch you.

    - Stretch just a bit, but within your adjacent possible.

  - A lot of secrets of success in a system is to keep it in its zone of proximal development as much as possible.

    - How do you maximize the amount of time the system is in its zone of proximal development?

    - How do you create your maximal learning environment?

  - A couple of examples:

    - Toddlers will lock into their zones of proximal development, do a task 10 times in a row until they master it.

      - As a parent you see this happen often, perhaps once a day, always on a different task.

      - They’re *locked in* at that moment, totally focused on repeating the task.

      - But never for a task they were told to do, always for a thing they decided to do on their own.

    - Coevolutionary loops in adaptive systems are also a mutual zone of proximal development.

      - Coevolutionary loops give huge amounts of momentum.

      - Because you have an adversary that is well matched and you keep on trying to get an edge, which leads them to match and find a new edge, back and forth.

      - Constantly in the zone of proximal development.

      - One of the reasons Generative Adversarial Networks work so well.

- Asking "how do you develop taste" reveals that you don't have it.

  - If you believe that you're cool and you act like it, and other people agree, then you are cool.

  - Rick Ruben says "you've got to have confidence in your taste"

  - You can discover your taste by seeing what's different and resonates with others, and then sharpen it.

  - Discovering your taste is seeing what others resonate with and glomming onto it.

- Dialogue helps you discover and intensify the starting taste.

  - It’s a coevolutionary process with both sides of the conversation moving the idea forward.

  - A dialogue can create better results, even if it’s a dialogue between two “boring” thinkers.

  - LLMs can have more conversations, even with other LLMs, to find interesting non-centroid beliefs.

  - For example: have one LLM participant play the role of generating ad copy. Have another one play the role of a skeptical consumer reacting to the copy, in a loop.

  - They start off with boring centroid ideas, with just a bit of randomness.

  - Then the "dialogue" helps intensify the random noise into the fullest, most resilient formulation of itself.

  - The LLM distribution stays the same (the model doesn't change); the context drives them off-centroid.

  - The generator/critic pattern.

  - Generate a bunch of stuff, select, and then amplify the ones that work.

- People assume that things with numbers are inherently objective.

  - But that’s not true!

  - Tons of hidden assumptions have to be embedded in the decision of what to measure in the first place.

  - For example, polling for elections has to make adjusting assumptions about response rates of different subpopulations, etc.

  - The results look objectively true, but are actually largely created based on the baseline assumptions of the model, what the electorate “should” look like based on what it’s looked like in the past.

  - But the underlying context that those assumptions are about could change invisibly to the model; it’s outside the model.

  - And then the numbers would be fundamentally wrong later.

- When the bar to get good enough is low, people are more creative.

  - When the skill bar is so high that you can't plausibly reach it you don't even try.

  - When the bar is low and you get little boosts of acknowledgement as you make progress, you *want* to be creative.

- Indie selects for generalists with taste.

  - True in film, games, art, etc.

  - If you're doing any of those industries as part of a machine, you learn just how to turn that crank as part of a larger machine you likely don't understand at all.

  - But if you're doing indie, you have to have an approximate sense of all of the aspects of creating it.

  - Much richer knowhow.

  - To be successful in an indie context you must have taste.

- Execution vs creation.

  - Pragmatism vs idealism.

  - Convergence vs divergence.

  - Capping downside vs creating upside.

  - "Make it more the thing it already wants to be" vs "Make a new type of thing"

  - Neither is better.

  - You need both, in a proportion that varies by the context.

- We've contorted ourselves to how the software we use works.

  - Why not have the software we use contort itself to us?

  - In the past, software couldn't contort itself.

    - It’s a machine, not alive.

  - Now software can be squishy, organic, alive.

- Lots of use cases are properly calendars.

  - E.g. a TV Guide, or a workout plan.

    - Any time-series of blocks of content.

  - But the appification of everything means that most calendars are the most generic form of calendar.

    - A calendar that is good enough for everything and great for nothing.

    - The lowest common denominator calendar.

    - This is the defining character of an app-first approach.

  - What if you could have a calendar display perfectly suited to the particular type of data you were working with?

- Most software today starts with functionality, then adds data.

  - But you can flip it: make it data first, and then functionality appears.

  - Instead of getting one-size-fits-all apps for a large class of data, you get functionality perfectly situated to the particular data.

- Before, software you used exactly once made no sense.

  - Writing software was expensive!

  - Even if you were an amazingly competent engineer, it still took time and effort.

  - But now LLMs make writing quick code many, many orders of magnitude easier.

- Having all the data in one place was not enough before LLMs.

  - Because you still needed a hyper-knowledgeable engineer to design and *build* any bit of functionality.

  - Swarming emergent functionality of finely enmeshed gears is way harder to accomplish.

    - All of the gears made by experts need to enmesh with all of the other gears precisely.

    - A coordination problem on top of the heavy lifting of engineering!

  - But now with LLMs if you get all of the data in one place and give just the barest of scaffolding, useful functionality can just emerge in place.

- In a world where software is expensive to write and cheap to run, you get larger chunks of software.

  - The complexity of adding features to software or migrating the schema of the underlying data goes up with the square of the number of use cases.

  - When software becomes cheap to write, you get larger amounts of smaller bits of software.

    - Each bit of software is much easier to modify and tweak because it is smaller.

  - Need a new use case?

    - You don’t necessarily need to modify a bit of existing software and make it more complicated.

    - You might be able to add a simple small additional separate tool that interacts with the pre-existing pieces, without making the pre-existing pieces more complex.

- A typical architecture has point-to-point structured communication.

  - You want two components to talk? You add a specific API, carefully structured and bespoke to the purpose, for them to talk.

  - This works reliably… but also can’t create surprising, novel, emergent value.

  - As the system gets larger, you get a combinatorial explosion of precise, brittle, API wires holding it all together.

    - It gets more and more expensive to change anything fundamental, or to build a new kind of thing because it needs tie-ins with n^2 other things.

  - Dynamicland and other systems like blackboard systems try a different way.

  - If you have a single shared reactive database you can use it as a message bus.

  - Everything talks directly to the database and then gets updated reactively when something it cares about changes.

  - Instead of pairwise communication you get a hub and spoke structure.

  - It’s messier and sometimes one thing will change that breaks everything… but it also has the potential for emergent upside, for small bits of activity by a motivated user to unlock game-changing new value in the system.

- Data schemas are extremely high leverage in a world of LLMs.

  - LLMs given a rough schema for what data to keep track of in the application can do a great job generating code with only a small bit of english language prompting.

    - You can give the schema in any number of formats.

    - I find a simple Typescript type definition is the easiest.

  - This was one of the insights that emerged for me working on [<u>Code Sprouts</u>](https://github.com/jkomoros/code-sprouts) last year–with just a little attention to schema, amazing functionality emerged, almost automatically.

  - The schema defines the domain of what kinds of things the software will be able to model in the data and thus accomplish.

  - Once you have the schema the code is often quite simple.

  - Thinking in schemas is very natural for people with engineering experience.

    - It doesn’t feel like the *main* task in engineering today because there’s often a lot of code you have to write, at great expense.

    - But when writing code becomes easy and cheap and evaporates away, what is left is the centrality of the schema.

  - Thinking in schemas is extremely unnatural for people without engineering experience.

    - It’s an abstract task of generalizing.

  - A tool that allows users to express a schema and amazing things sprout out of it will make it easier for people to become LLM wizards.

    - And yet requiring the first step to be a schema will set a low ceiling on the number of users who can use it.

  - Luckily LLMs are pretty good at extracting a schema, too, if directed to do it.

    - LLMs are great at extracting a schema from a series of example bits of data.

      - A UX for users to collect bits of data they want to operate on, and then software sprouts out.

      - The first step in the LLM generation is to extract a schema automatically.

    - Another pattern: you can simply ask a user to define a few things they want to do, and the LLM can rough in a schema.

      - If the schema isn’t right, it can be easily modified and extended by the LLM for additional use cases.

      - One of the reasons defining a schema is hard is because you have to think forward to the types of use cases you’ll want to add in the future.

      - But when software is cheap, you can simply modify the schema when you want to add functionality that requires it.

        - Changing schemas used to be hard because you had to update all of the software that relied on it.

        - But if software is smaller and bespoke, the overhead is much less–the complexity of schema migration goes up with the square of the number of use cases.

        - And if you need to update the simple bit of software, simply pass it to an LLM and say “patch yourself” and it does.

      - Just-in-Time software with a Just-in-Time schema at its core.

- There are different apps for specific niches.

  - A general game? Use Unity.

  - A 2D game? Use Game Maker Studio.

  - A 2D RPG? Use RPG Maker.

  - The closer the tool is to your particular niche, the faster you’ll be able to do common things for that niche.

  - But there’s a tradeoff: it will now be *harder* to do things that don’t fit the niche.

  - You pick a tool at the start of your use case.

    - It’s a scaffolding, like a jungle gym, allowing you to reach far higher than you could alone.

  - But as your use case grows and becomes unlike the tool, it gets harder.

    - The jungle gym becomes a cage, preventing you from reaching in directions it didn’t anticipate.

  - But in a world of Just-in-Time software, this problem goes away.

    - You get on-demand software that perfectly fits your problem right at this moment.

    - The software doesn’t need to be complicated and prepare for any possible need, it can just respond to exactly what you need right now.

    - As simple as possible, but no simpler.

  - We’re so used to hardened tools.

    - Software is the ultimate tool. Hard. Extremely high leverage.

    - But soft tools are entirely different.

    - LLMs allow the creation of soft tools.

- A team of researchers in 2017 [<u>tried to estimate the value of different services</u>](https://www.pnas.org/doi/10.1073/pnas.1815663116).

  - They asked them how much they’d have to be paid to give up access to categories of tools.

  - The results:

  - search: \$17,530 / year

  - email: \$6,139 / year

  - maps: \$2,693 / year

  - video streaming: \$991 / year

  - social media: \$205 / year

  - Significant consumer surplus!

- Escape hatches allow an open-ended system to absorb many more use cases.

  - When you add an escape hatch, many use cases go from impossible-no-matter-how-much-effort to possible-with-significant-effort.

  - But a key danger: now you’ve taken the pressure off your system, and you forget to add higher leverage ways of doing those newly enabled use cases.

  - “Simply use the escape hatch!” you say to any new use case.

    - But use cases that use the escape hatch are a form of debt.

    - Yes, the creator of the feature was able to accomplish that use case, but their effort doesn’t help any other creators with similar use cases.

    - Each use case has to do a lot of effort on its own.

  - The escape hatch takes the pressure off and allows you to peek at the actual use cases people have so you can learn from them and prioritize based on usage.

  - But now you have to sublimate those escape hatch use cases, to make platform improvements to the non-escape-hatch portions, making it easier for creators to make things with less effort.

  - Code gets stronger when it’s exercised, and you want use cases to use the actual platform primitives, not take the path of (individual) least resistance and use the escape hatch.

  - So make sure to continuously be trying to minimize the number of real-world use cases that *must* use the escape hatch.

  - This forces you to constantly improve the platform in ways that will allow even low-motivation creators to create more and more powerful use cases with less and less work.

- Game engines create the potential for amazing, complex experiences.

  - A diamond hard, efficient, resilient underlying game engine.

  - And then most of a game’s logic is implemented in squishy, easy-to-prototype logic like Lua.

  - A hardened foundation for open-ended creativity on top.

  - A good two-layer shape for many open-ended generative systems.

  - Web development: diamond hard runtime of the browser, soft and squishy web apps on top.

- Email is the decentralized, ubiquitous social protocol that already exists!

  - A classic from Gordon: [<u>Everything Talks Email</u>](https://newsletter.squishy.computer/p/everything-talks-email)

  - Second Life allowed you to send an email to any object in the game.

    - Objects could also send emails *out*.

    - But if you wanted those messages to cause something on the other side to *do* something in response, the message had to be formatted exactly correct to work with the plain old code on the other side.

    - But imagine if you had an LLM on the other side!

    - Like forwarding to a person and knowing they'll do something reasonable.

- LLMs can’t write particularly large amounts of code before they start getting confused.

  - They can do maybe 1k lines of code before they start losing track of how the whole thing wires together.

  - This will presumably get better as larger context windows get more resilient, but presumably LLMs will always have a steeper dropoff in ability for large code bases.

  - But you can set up the system to still get maximum leverage out of that ability.

  - For example, instead of having it write components from scratch, give it high-quality, useful components to wire together in novel ways.

    - And define clear interfaces where other code can magically supply the necessary inputs from elsewhere in the system.

- The creator mindset is different from the buyer mindset.

  - The creator mindset is more an owner mindset than a renter mindset.

  - You feel more deeply intertwined with the thing when you help create it.

  - You feel more invested in it, more willing to continue to invest vs throw it out.

  - Products can lean into this feeling of creating: the IKEA effect.

    - Apparently when ready made cake mixes came out, they originally didn’t require an egg, just add water, mix, and bake.

      - But that felt too simple, like the baker wasn’t creating, it was almost cheating.

      - So they had you add an egg, which made it feel just enough like creating.

      - Claude tells me this story is likely apocryphal!

- LLMs make the “good enough” viable zone larger.

  - A product has to help users get to a “good enough” result quickly for it to be viable.

    - Before the user gets to a good enough result they are liable to give up at any moment and never return.

  - Many of the most powerful tools have a steep learning curve before they can output anything good enough.

    - That means they’re only viable with a small, hyper-motivated market.

  - Once you have a viable product (it has achieved PMF with some small audience) you can move to hill climbing.

  - How much you have to plan and test before shipping is tied to how easy the “good enough” target is to hit for users, in a way that stands out from others.

    - For traditional software in a crowded area the target is very small.

  - LLMs allow much more forgiving tools that can get you a good enough result quickly.

- LLMs allow you to go from "random idea" to "a thing that vaguely works" way, way faster.

  - This is the crucial phase where most ideas die.

  - The time between “huh, maybe…” to “this kind of works!”

  - Once an idea gets to viable, it coheres on its own; choosing to incrementally extend it makes sense, and you are more likely to naturally do it.

  - Before that point, every incremental unit of time has to go into a thing that will likely never work and might be wasted effort.

    - So you’re less likely to do it.

  - You effectively have to cross the chasm, from a non-viable idea to a viable one.

  - But LLMs, for lots of simple projects, allow you to leap over the chasm.

  - If everyone could be a wizard with LLMs, we’d have way more viable ideas as society to build on.

- Engineering vs product approaches come at problems from different angles.

  - Engineering is hard; it tends to start from the bottom and accrete hard, well-defined layers upwards.

  - Product is soft; it tends to start from the top, from an approximate whole, to find resonance, and then harden and flesh out the parts that are working.

  - The danger of the engineering mindset is that you run out of runway before building anything resonant enough to be viable.

  - The danger of the product/design approach is that you design something that cannot be built for real.

    - A sketch of a castle in the sky.

    - So to avoid that you go with only off-the-shelf tech that will definitely work, and to get something different you go for a novel *combination*.

  - If you're trying to innovate on possibility and on resonance at the same time, it's hard!

- When a thing is viable and coherent it self coheres.

  - Until that point it has to have someone input a huge amount of continuous effort and force of will until it gets to that point.

  - Default cohering vs default decohering.

    - A seemingly small difference that is actually an infinite difference.

    - Does each incremental bit of effort around the thing build it up or diffuse/erode it?

    - Most things in the world are default-decohering.

  - How viable and compelling is the core vision of the thing?

  - How plausibly manifested, so everyone who can see it sees it should exist and how to make it a more fully realized version of itself.

- Things pop from default decohering to default cohering when they become a thing people believe is viable and valuable.

  - Viable = Actually works

  - Valuable = Actually worth doing

  - The viable and valuable center becomes a thing that people nearby want to extend and improve.

  - Before you have that core, every given bit of effort or change is equal.

    - Who’s to say which is better?

    - So the incremental actions pull it the whole in random directions, decohering.

  - But when you have a core that collaborators believe is viable and valuable, there’s a default thing to work on.

    - You could choose to improve the thing that is working, or a different way (which might turn out to not work).

    - The thing that is an incremental improvement to the way that is working is, all else equal, obviously better.

    - This asymmetry is what creates a default cohering thing.

  - For this to happen, a critical mass of collaborators have to believe that it is viable and valuable.

    - You can believe it because it’s self-evident, you can see it with your own eyes.

    - Or you can believe it because you believe the vision or feel morally connected to the collective whose goal it is.

- Two kinds of energy in a team: convergent energy vs individually brilliant energy.

  - Convergent energy: each bit of energy steelmans and improves the current centroid of the system.

    - Makes the system a better, higher fidelity realization of itself.

    - A “yes, and” energy that builds on the thing in front of them and helps make it better.

  - Individually brilliant energy: Someone with innovative ideas about how to knock the system off its centroid, to pull it in a new direction.

    - This energy diverges the system… but possibly in a direction that will turn out to be much better.

      - In contrast to randomizing energy, that pulls the system off its centroid… but in a random direction, not an improvement.

      - To see a better path than the one it’s on that could be much different requires a particular kind of brilliant mind.

    - Note that two different vectors of individually brilliant energy can clash or counteract each other.

  - You need a mix of both of the energies, with the ideal proportion shifting at different times.

    - If it's a collection of brilliant people then it's just a random self-eroding swarm that never coheres.

    - If it's all convergent energy you converge on mush.

  - A key complement to these energies: a curation function.

    - Look at all of the individually brilliant ideas and select the subset to be adopted into the main thing.

    - Instead of just random bottom-up energy, you have a curation function that can be coherent to a vision.

  - Different people bring different energies naturally.

  - When you combine all three of brilliant energy, a curation function, and convergent energy, you can get a sublime result.

- Without a coherent, plausible, inspiring vision, a team default-diverges.

  - People aren't diverging on purpose, they just don't know what the vision is.

  - The path of least resistance for a bit of work will go in some random direction.

  - As everyone on the team follows their individual path of least resistance, everything tends to decohere.

  - But a plausible vision is like a magnetic force; things pull towards it unless they have a strong reason not to.

  - It becomes default-cohering.

- Once a project is default-cohering, you can hill-climb to improve it.

  - You get ever more precise requirements, and get ever better and higher fidelity implementations of the requirements.

  - But this execution playbook cannot find new hills.

  - You need to have a burst of creativity to create a new kind of thing.

    - At the beginning it will be rough, messy.

    - But if it gets to the point of being viable it becomes default-cohering.

    - Then past that point you can simply optimize it to be a better version of itself.

- When you optimize you decimate hidden reservoirs of adaptability.

  - Optimizing is about making the system more the way it already is.

    - Just better, faster, more efficient.

  - The value is direct and concrete.

  - The danger is indirect and speculative.

  - But often the clear value is an order of magnitude smaller than the indirect danger.

  - But no one ever got fired for optimizing the system as it is.

- In each time step, take a little time trying out a few different safe-to-fail things that you have intrinsic motivation on, and spend most of your time leaning into the existing things that are resonating.

  - That's it!

  - A general purpose algorithm for most things in life.

- The more that people actually keep the supposedly canonical thing up to date with reality, the more it actually becomes canonical.

  - For example, a kanban board of tasks on a project.

  - A canonical thing needs to be default-convergent.

  - Close enough to reality that the incremental action people take on it brings it *closer* to modeling reality.

  - But a supposedly canonical thing that doesn’t represent reality is default-divergent.

    - No one will want to clean it up (each incremental bit of improvement doesn’t make a dent; it’s still not representative of the underlying reality).

    - So as time goes on, and reality continues evolving, it moves further and further away from the supposedly canonical representation.

    - Now you have an “official” thing that doesn’t actually represent reality… confusing!

    - At a certain point you just have to call bankruptcy.

- The fact that you can have different thoughts than others is the reason interesting thoughts exist at all.

  - If you had to have the same thoughts as everyone else, nothing interesting, nothing innovative could happen.

    - It would all just be pulled to the centroid, the status quo.

  - And yet the boundary between our minds is hard to communicate over, and why sharing a good idea is so hard.

  - The vast majority of human experience is just running around in a cacophony of people shouting to be heard with what they think is a useful insight, above the cacophony of everyone else doing the same.

  - "What's that??? I can't hear you, I'm so busy sorting through all of the incoming signals yelling in my ear about things that don't matter and I don't care about!"

- A funny thing about knowhow: the more you develop it, the more invisible it gets to you.

  - It just becomes automatic, something you don't even have to think about, which makes it hard to interrogate and inspect.

  - That means it's hard to teach, and sometimes you erroneously conclude there's not even anything to teach.

  - "Simply do it, it's not hard!"

  - This is the curse of knowhow!

- Steering your focus is how you think.

  - Focus is hard to steer!

  - It kind of steers itself often and you have to wrestle with it to keep you aligned with where you want to go.

  - The more intentionally you can steer it, the more effective you are.

- System 1 is automatic.

  - No conscious thought or effort.

  - Consciousness is about focus/attention.

  - Focus is about where you steer your System 2.

- Feynman believed you don't *truly* understand a thing unless you can explain it in three fundamentally different ways.

  - Sometimes experts cling very heavily to the way they were taught it.

  - Holding it too tightly, focusing on superficial ripples, not fundamental undercurrents.

  - It takes time to be ready to absorb atypical ways of understanding something… but when you do it often feels like an epiphany, like you see the underlying hidden order of the system.

- A very different stance: "You convince me" vs "I'm going to try to convince myself."

  - Default diverging vs default converging.

  - Do you believe being coherent with the whole that you are part of to be inherently valuable?

- In a discussion, who's willing to entertain the idea the other person might be right?

  - People think, "well I'm smarter than them in this dimension, and they're wrong, so entertaining they might be right would lead to the wrong conclusion."

  - But what if the other person is right in some dimension you don't see?

  - Perhaps you're looking at the discussion in one dimension where they appear to be wrong, but actually they have a more holistic understanding of the other relevant dimensions, and those dimensions are hidden to you.

- Naming is hard, mainly because you have to agree on what subset of things are being named.

  - It appears hard because everyone has different taste on what specific word to use.

  - But that's a superficial distraction, obscuring the actual disagreement.

  - Everything is a borderless gray goo, and you need to figure out where to draw a border.

  - A name gives a shared handle to a concept.

    - But more important than the handle is what things it is attached to.

    - That is, what subset of gray goo is inside the concept’s borders.

  - The border is fully implied, indirect, hard to see, but it matters the most in a name.

  - If you draw a border that is very much like another thing and adopt its name, you now can't change the border.

    - You draft off previous understanding, but in a way that also brings in semantics you didn't mean to include and are now confusing.

    - The pre-existing momentum helps people understand the name from the beginning, but also makes it harder to change.

  - A new name that is related but distinct allows you to formalize the border of what's in and out later and tweak it… but doesn’t come with the pre-existing awareness of the name.

- If it succeeds, great.

  - If it fails quickly, great.

  - If it fails slowly, it's an excruciating waste of time.

- You could win the lottery... but it's safer to assume you don't.

  - If you assume you'll win the lottery you'll almost certainly fail.

  - If you assume you won’t win the lottery you’re less likely to die.

    - And there’s still upside if you do turn out to win the lottery!

- if you have no edge then your only edge is execution.

  - Stumble and you will be overtaken.

  - A desperate, paranoid existence.

  - Strategy is all about finding a non-incremental, and ideally compounding, edge.

- Complex adaptive systems can’t be understood just at the agent or the collective level.

  - For example, the bee vs the hive.

  - If you focus on just optimizing one level you’ll create maladaptive outcomes in the other layer.

  - The system emerges from the *interplay* of those two levels

    - Note that this agent vs collective interplay nests, fractally, down to the very bottom of the system.

  - The hallmark of complex systems is you have to optimize the *whole*.

    - If you optimize any one part then you’ll create weird problems in other parts of the system.

- Take risks you can afford to lose.

  - That is, that won’t kill you.

  - People with existing resources are less likely to be killed, all else equal, by the same risk.

    - They can absorb more downside without it bankrupting them.

    - A structural advantage.

  - Risk has upside, but also downside.

    - A downside that won't kill you caps the downside, leaving the uncapped upside.

  - Capped downside, uncapped upside.

  - The more the asymmetry, the more you should just do it, get as many spins of the roulette wheel as possible.

- A measure of engagement on a team: what percentage of people show up to the all hands.

- A great way to make a team feel like a team: SWAG.

  - Helps the mindset go from “me” to “we”.

  - Because teams are virtual objects, not physical.

    - That's doubly so for remote teams.

  - So lean into the physicality, the totems of shared identity.

  - A physical object that everyone on the team gets one of, and is proud to get.

- The most interesting things are the hardest to compress.

  - Interesting things don't fit in the surrounding system, fundamentally.

    - If they fit into the status quo, if they went with the grain, they wouldn’t be interesting.

  - Interesting things are about innovation, but also they are dangerous to the existing order.

  - The existing order will preserve itself, even if no individual in the existing order intends to preserve it or thinks they care that much.

- Organizations optimize for order, not for truth.

  - Because if it goes against the grain, it's potentially dangerous.

    - It could erode the hard-won order and structure.

    - Or even cause an explosion that throws everything into chaos.

  - There’s a reason there’s a grain: the grain *works*.

    - It has at least worked in the past, and is known to be viable.

    - Going against the grain has to prove not only that it is valuable but that it is viable.

  - Imagine a 2x2 of correct and going with the grain.

    - Going with the grain and correct: Extremely easy to do.

      - Everyone agrees it’s important and will celebrate you for doing it.

    - Going with the grain and incorrect: No one ever got fired for doing it.

      - It didn’t end up working, but at least you didn’t upset the apple cart.

      - You’re very unlikely to get punished for it.

    - Going against the grain and incorrect: Everyone can agree to not do.

      - Dangerous and value destructive.

    - Going against the grain and correct: Might be game-changing, but very dangerous to attempt.

      - Because it goes against the grain, if it turns out to work, it could change the system, possibly in a game-changing way.

      - But also the organization will fight it strongly, because it goes against the grain.

      - The bar for convincing other people it’s not dangerous and is worth doing is much, much harder.

      - If you fail, you’ll get punished and maybe even knocked out of the game.

  - As an organization you want to encourage some of that last bucket.

  - But as an individual, it’s significantly more downside than upside to try an against-the-grain idea.

  - So the emergent equilibrium is individuals in the organization working to maintain order, not truth.

- Execution and creation are different.

  - When it’s a project that is your vision, you get to exercise both muscles.

  - When it’s someone else’s vision, you only get good at execution.

  - The machine doesn’t want you to have your own vision.

    - That creates chaos when everyone does it.

  - The machine only wants you to be good at execution.

  - You have to be the one to want to be creative.

- Inside of an organization, the game, the kayfabe, will draw all of your attention.

  - It will make you feel like it's what matters, when in reality it's the emergent game that sucks up all the extra energy.

  - If you see that the kayfabe is an illusion, a distraction, a thing to hold at arm's length and only do the bare minimum then it's fine.

  - But sometimes people get confused and see it as an end in and of itself.

  - If you're good at playing the game, the org will reward you for it, and it will become part of your ego.

  - "I'm good at this, and this is important," will be what the org will whisper in your ear.

  - And when you believe those whispers, you are lost.

- If you fight against kayfabe, you will lose.

  - Kayfabe is a dysfunctional extreme version of organizations optimizing for order over truth.

  - The reason you can't correct kayfabe and "fix it" is because it's an emergent, self-catalyzing phenomenon.

  - It has orders of magnitude more energy than you.

  - It's like gravity.

  - No individual has enough strength or patience to fight it successfully.

  - It will win.

  - In an environment infected by kayfabe, you have a choice to make: submit or exit.

- We're embedded in lots of systems all of the time.

  - How adaptive and organic the system is is how much of a “machine” it is.

  - The more of a machine, the more inhuman.

  - The machine you work at, the one who pays your salary (and claims a monopoly on your productive output) is a particularly important and dominant force.

  - You aren't stuck there forever (you aren't a slave), but you can only be at one at a time, so switching is hard.

- Being trapped inside of someone else's bureaucracy is a special form of hell.

  - But if it’s a bureaucracy you helped create or that is advancing a cause you deeply believe in it’s less bad.

- Things in the world by themselves don’t have meaning.

  - When a human touches them the meaning is created.

  - The choice to touch it gives it meaning.

  - Meaning is about cost and choice.

- The seeds of meaning are embedded in everything around you.

  - You just have to choose to see them.

  - To allow them to grow.

- I’m about to tell you a secret that will either cause you to roll your eyes or will make you feel transcendent and complete.

  - What’s the difference? What determines which you feel?

  - You.

  - Are you ready for it? Are you open to it?

- The meaning of life is *this*.

  - Not *that*. *This*.

  - Be here now.

  - Every moment is a glorious struggle of challenge and growth.

  - You don’t *have* to, you *get* to.

  - The journey is the meaning.

  - Why not make *this* meaningful?