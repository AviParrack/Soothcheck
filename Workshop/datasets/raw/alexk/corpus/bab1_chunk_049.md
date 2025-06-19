# 12/4/23

- Imagine looking at an existing, heavily used system.

  - One mindset is to look at it and see all of its flaws: how messy it is, how imperfect, how hard it is to change or improve it.

  - Another mindset is to look at it and see not what it is but what it could become.

    - To see how it's beautiful that it is so heavily used, what a privilege that is to get to work on a thing that has so much leverage.

    - How with humble, continuous, long-term-oriented clean-up work, you can help it become something far, far greater than what it is today.

- Choosing to share metrics externally is a big step.

  - It's not just a moment-in-time decision; it's ongoing.

  - For the foreseeable future you'll get "In Dec of 2019 you reported X was Y. How has it changed since then?" kinds of questions.

  - If you *don't* answer those questions (or share the updated data proactively) people will assume that it's because the metrics look bad.

  - This is true for each and every individual metric you choose to share.

  - If you share the same metric multiple times on a given implied cadence, it becomes super-linearly hard to avoid breaking the precedent of sharing.

    - When you stop sharing, people will assume for sure that it's because the numbers aren't impressive, in proportion to how established the precedent for sharing is.

  - A counter-intuitive best practice for sharing a new metric: don't share it again in the time period after it was first shared, to avoid setting a precedent.

  - Also, share it only at random, hard-to-predict intervals, making it less obvious when you stop sharing it, and less likely to get "the metrics must be bad" suppositions.

    - (Of course, if you stop sharing it for good, at some point that will be obvious, no matter how random the interval... it will just take longer for people to notice.)

- At a developer-focused company, which is more fundamental: the API, or the UX (e.g. a dashboard)?

  - The API is the escape hatch; everything must be possible in it.

  - But APIs are fiddly and require specialized knowledge; there are many more people who would feel comfortable with the UX than the API.

  - UXes are more forgiving and allow more users to discover and explore.

  - Ideally you make as much possible with the API, while simultaneously making it so people can achieve as many goals in the UX and don't *have* to go to the API unless they want to.

    - You can think of this as a special case of the "geek mode" pattern from [<u>doorbells in the jungle</u>](https://komoroske.com/doorbell-in-the-jungle).

- Built things vs living things have radically different properties.

  - A built thing, like a building, has to be designed by someone to be able to carry a load.

  - It needs someone else to carry in their mind a complete conception of it that is viable in the real world.

  - A living thing like a tree can sense where a branch is swaying and build up more lignin there to help resist the sway.

  - The tree doesn't even need to have a full conception of itself, it just does the obvious thing in the local situated sub-context, and a coherent whole emerges, automatically.

  - Living things adapt for free. Built things cannot adapt themselves.

- UX changes are more forgiving than API changes.

  - A few weeks ago I mentioned that when you change a UX you change the interface, meaning more users might notice/be confused (because the change will at least show up in people's peripheral vision if it's on a top-level screen, whereas a user of an API might not notice at all other parts of the API changing)

  - However, there's a countervailing factor:

    - UX is for humans, and humans are alive.

    - APIs are for code, and code is built.

  - Yes, a human wrote the code (probably) but once the code is written, it might not be changed for a long time even while it executes often.

    - If you were to sample from all code being executed across the universe at any given time, the vast majority of code would have had orders of more time of accumulated execution wall-clock time than the amount of wall-clock time of humans thinking about it when they wrote it in the first place.

    - This is part of the extraordinary leverage of code.

    - (There is code for which this is not true; e.g. little random glue code, but it wouldn't be prevalent in your sample because it's rarely executed)

  - Code can't adapt itself to integrate with an API; a human can adapt themselves for a changed UX interface.

  - There's an entity with agency at each time step of interaction; the human is always in the loop with UX, while with code the human is often only indirectly / rarely in the loop.

- It’s hard to understand how a black box works.

  - You understand a system when your mental model can accurately predict what it will do in response to most inputs.

  - With a black box you don't know what state it’s in at any given point and have to guess.

    - And as more things happen it drifts away from the previous state you thought it was, making it very hard to "catch up" your understanding of its state to what it's actually in.

  - A non black box is constantly releasing signals about what state it is, helping you synchronize your mental model to its current state.

  - Two things that make it hard to synchronize your mental model with how a system actually works:

    - 1\) A complex internal state (how many distinct states can it be in?)

      - You need to distinguish the current state from the other states it might be in, a task that scales in difficulty based on the number of possible states.

      - This is even harder when there are some states that are very rare, meaning even someone who has interacted with it a lot is unlikely to have experienced it.

    - 2\) Very few externally-visible signals of what state it's in.

      - The extreme of this would be a complex device with hundreds of input sensors, thousands of possible internal states, and only a single LED output.

      - (You could imagine an instruction booklet for a device as being a part of the signal that comes along with it as a complement.)

  - If you have lots of external signals then over time with enough experience users can build up a (perhaps intuitive, flawed) understanding of the internal state.

  - There's basically no way to bootstrap understanding of a system with very few external signals.

- Caring deeply about quality requires making tradeoffs.

  - E.g. investing significantly in expensive QA, doing the continuous refactoring work to clean up a jenga tower to make it into a pyramid, or pivoting much less often.

  - If you don't make a tradeoff, then you're cosplaying at quality. You're just doing the superficial components: the optics, not the fundamentals.

- If you set grand plans for a thing that will be great once you complete it, do step 1, and then get distracted and don't do the rest of the work you planned to do, you might end up in a worse situation than having never taken the first step.

  - Plans that are good ideas even if you don't do anything else are much better than plans that require executing multiple steps before it's a good idea.

- Everything by default diffuses.

  - This is the second law of thermodynamics: entropy.

  - Diffusion creates variability / diversity, which is the necessary ingredient of innovation (more likely to have exposure to what turns out to be the right idea).

  - But to create something that coheres and endures, it often requires coordinating multiple people at the same time to build it.

  - One way to do this is focus: "this is the thing we are doing right now, ignore everything else".

  - Focus can help build things (and ultimately nothing matters if it doesn't happen in the real world); but it reduces diversity and thus resilience.

    - What if the right idea to survive or thrive is not in the region you're focusing on?

  - Given this, how can you make good things accumulate in the real world?

  - There are two complementary approaches: breadthwise and depthwise.

  - Breadthwise would be configuring it so the various actors don't *have* to coordinate to accrete things of value.

    - The non-coordinating group of actors is a swarm.

    - This is one of the benefits of ecosystems; the swarm finds and builds many random things, and the good things hang around while the bad things erode away.

    - The overall cost of the effort of the swarm is massive, but for any single participant (especially one who benefits no matter who in the swarm finds it) it's minimal.

  - Depthwise would be slicing your actions into smaller actions.

    - Instead of doing one big thing that takes a lot of time, do it as a series of smaller steps, where you can adapt in between steps.

    - This helps your OODA loop tighten, giving you diversity/adaptability while allowing instantaneous focus.

    - Of course, this isn't a panacea:

      - Sometimes the administrative burden of many small pebbles is much higher than one big boulder.

      - The process of sharing information / deciding on the next step (coordination) has super-linear costs.

      - So sometimes it's best to pay the coordination cost once for a big boulder, but at the cost of adaptability / resilience.

      - This "small number of pivots" logic is the "big rig" style of organizations.

- In front of each of us is a wave function of possibility: things we could do.

  - Agency comes from the act of collapsing the wave function of things you *could* do, to the thing you *did* do.

  - If you take a longer-term perspective, you can see that your actual situated wave function is a broader, partially-collapsed wave function based on decisions you made in the past.

  - A visual: your decisions right this moment are a zipper of possibility. You can steer the zipper through space as you go forward, but behind you is fully zippered into one timeline.

- We do not have infinite agency.

  - Our surrounding context constrains our possibility space.

  - That context is a complex emergent cacophony of physical processes, but most importantly the decisions of everyone else we interact (directly or indirectly) with.

  - You can only hold the idea "every decision I might make is equally good" if you don't interact with the rest of the world.

  - When things interact with the real world, they need to be ground-truthed: shown to be compatible with the world as it exists.

  - Interactions with other people open up the possibility space (you can do more together than what you could do individually), while limiting your agency within it (requiring you to cooperate).

  - Choosing to cooperate or interoperate with others means harnessing the power of the whole at the expense of your local power to choose implementation details.

  - Arendt defines agency as ~"non-undoable action taken within a community of other individuals"

  - We are all embedded in a fabric of interdependence way larger than we can comprehend.

- People will take more risks in trying out things when they know they get to play the game multiple times.

  - If you have one shot, you'll do the safe thing.

  - This is the thinking behind the tit-for-tat prisoner's dilemma strategy. "I'll take the risk of cooperating in the first round, but only because we have other rounds if it doesn't work out."

- When we expect to make a decision, we are in a massively different head space.

  - When we are passively watching something, our brains can wander with no real downside.

  - But when we expect to have to make a decision (e.g. in a game, or when executing in our job), we are constantly absorbing all relevant information, balancing across a lot of possibilities, developing an intuitive predictive model, and being ready to have a decision if someone said "OK, go time, make your decision RIGHT NOW".

  - The two situations look somewhat similar externally but are radically different inside the mind.

  - The latter has orders of magnitude higher rate of knowhow acquisition: much faster learning rate.

  - This is why simulations, games, and play are so good for learning.

- The most interesting things come from straddling a boundary.

  - A boundary separates nearby things with a discontinuity.

  - The things on either side of the boundary are close by but discontinuously different, which means mixing them gives you ingredients to create something new, instead of mixing things that are mostly the same.

- Thinking long term takes practice.

  - Just like participating in democracy takes practice, and why de Toqueville's idea of democracy in the small as being important to support democracy in the large.

  - The more rarely you're in a situation where you have to think long-term the harder it will be to think long-term in other situations.

  - A randoms set of things that make it hard to think long-term:

    - A high expected rate of reorgs within an organization

    - Not having a set, consistent physical space to work in (you feel more like a nomad, and can't accumulate meaningful physical things in that space to aid your memory).

  - An observation: at a society level, the median time employees spend in a given job is declining.

- A few extra Generative AI predictions (because why not).

  - There will be an explosion of spaghetti code in the wild.

    - Spaghetti code will get easier to create by more people at much faster rates than our ability to understand / disentangle it will grow.

  - Tactics like scenario planning, simulations, prediction markets, Wardley mapping etc will become more prevalent.

    - These tactics have been known to be highly effective for a long time, but they are expensive and require convincing a skeptical audience to invest time in them.

    - But now they can be orders of magnitude cheaper when you [<u>can ask LLMs to do a lot of the work for you</u>](https://www.linkedin.com/posts/simonwardley_creating-a-draft-wardley-map-by-having-a-activity-7136088798500720640-gy-C/).

- I really enjoyed this piece from David Brooks: [<u>The Essential Skills for Being Human</u>](https://www.nytimes.com/2023/10/19/opinion/social-skills-connection.html). A few quotes that stuck with me:

  - "In any collection of humans, there are diminishers and there are illuminators. Diminishers are so into themselves, they make others feel insignificant. They stereotype and label. If they learn one thing about you, they proceed to make a series of assumptions about who you must be."

  - "Illuminators, on the other hand, have a persistent curiosity about other people. They have been trained or have trained themselves in the craft of understanding others. They know how to ask the right questions at the right times — so that they can see things, at least a bit, from another’s point of view. They shine the brightness of their care on people and make them feel bigger, respected, lit up."

  - "If we are going to accompany someone well, we need to abandon the efficiency mind-set. We need to take our time and simply delight in another person’s way of being. I know a couple who treasure friends who are what they call “lingerable.” These are the sorts of people who are just great company, who turn conversation into a form of play and encourage you to be yourself. It’s a great talent, to be lingerable."

  - "The really good confidants — the people we go to when we are troubled — are more like coaches than philosopher kings. They take in your story, accept it, but prod you to clarify what it is you really want, or to name the baggage you left out of your clean tale. They’re not here to fix you; they are here simply to help you edit your story so that it’s more honest and accurate. They’re here to call you by name, as beloved. They see who you are becoming before you do and provide you with a reputation you can then go live into."