# 11/25/24

- One of LLMs’ primary superpowers: they have human-level ability, but they never get bored.

  - For example, human engineers get bored very quickly writing tests for their code.

  - But having tests makes it significantly easier to innovate more quickly, because you can more easily tell if you broke something.

    - This allows a faster feedback loop.

  - Even imperfect tests are better than no tests.

    - As long as the tests are generally green, so you don’t develop a boy-who-cried-wolf insensitivity to flaky tests.

  - LLMs can generate tests from your spec for your program happily.

  - Look for opportunities that require human-level reasoning, but where real humans get bored quickly.

- Because humans get bored easily, information has to be wrapped in charismatic, attention-grabbing wrappers.

  - It’s much easier for our brains to grok a charismatic YouTube video than a dry page of technical documentation.

    - If we get bored or distracted or impatient, we give up.

  - So most content aimed at humans is *primarily* composed of fluff or filler, sugar to make the bitter information content go down smoother.

  - But LLMs don’t give up, they can plow through even the driest material.

  - That means that content written for LLM comprehension can be “compressed” *significantly*.

    - So boring no human would ever bother reading it.

- The power of infrastructure is it allows you to skip the boring parts at the bottom and focus on the innovation at the top.

  - Infrastructure creates its strategic leverage fundamentally by implementing a boring lower-level piece of functionality once, but allowing it to be shared by multiple users that otherwise would have had to implement it themselves, reducing the overall cost in the system.

  - LLMs can take the most boring bottom parts of tasks for humans, giving us leverage not unlike infrastructure.

  - But now that the boring parts are taken care of, you have to decide to innovate.

  - One option is to say, "Great, the computer already thought for me, I don't have to think. I wonder what was added on TikTok since the last time I checked it 30 seconds ago?"

    - A more passive stance.

  - Another option is to say, "Great, now that I don't have to waste mental capacity on the boring parts, what new heights can I reach? How can I think harder and further than before?"

- We use “frog DNA” to fill in gaps.

  - When confronted with ambiguity, we use our pre-existing knowledge of the world to guess at the resolution to the ambiguity.

  - Like in Jurassic Park where the dinosaur DNA has gaps that they fill with frog DNA.

  - LLMs have broad world-knowledge to draw from to fill in gaps.

  - Your prompt has to share all of the out-of-distribution specifics for the LLM to get the specific background and not just give you mostly a frog DNA answer.

    - If the thing you're trying to do is in-distribution the prompt can be small and draft off the existing understanding.

- Frog DNA is average, mush, generic.

  - That’s one of the reasons LLMs pull everything to the bland centroid; they fill any ambiguities with mush.

  - What if you could make it so instead of using frog DNA, it used *shark* DNA?

  - The right prompting or context-selecting procedures could help here.

- English specs can now be “compiled” to runnable code by LLMs.

  - When we make programs, we compile the source code down to an executable binary.

  - But then we make sure to keep the source code, so when we tweak it in the future we can compile a new binary.

    - Or, when we get better at compiling, or want to target a different environment, we can recompile.

  - The source code is far more precious than the binary.

  - But with “LLM compiled” specs to code, we currently keep the “binary” (the source code) and throw out the spec.

  - That seems backward!

  - This implies that LLM-native software development will place much more emphasis on the spec / PRD / design doc than the code.

- LLMs are a kind of cultural technology.

  - Markets are the original cultural technology.

    - Markets take a complex multi dimensional problem of allocating resources in society and reduce it to a simple signal to coordinate around.

    - Markets are an insanely powerful coordination mechanism.

  - LLMs are great at taking multi-dimensional insights from across society and reducing it into an object that anyone can talk to.

    - Instead of a one-size-fits-none piece of content (too intimidating for non-experts, too boring for experts), you get an organic, just-right artifact, because it can respond and grow to a particular user’s queries and follow-ups.

- Having a background in things like sociology gives a leg up for understanding LLMs.

  - Most tech is understood best with a straightforward math/computer science frame.

  - But LLMs are a *cultural* technology.

  - They are best understood through the lens of culture and society, not how they work.

  - The chatbot UI is what you get when you ask programmers to guess how users should interact with this new cultural technology

    - It’s effectively a command line, one of the most intimidating UIs you can imagine.

- Alison Gopnik has a wonderful metaphor: [<u>stone soup AI</u>](https://simons.berkeley.edu/news/stone-soup-ai).

  - In the stone soup story, a traveler seeks to make a soup out of only a stone.

  - He says that a small garnish will make it taste much better, and their neighbor obliges.

  - The garnish is minor compared to the existing quality of the soup so it makes sense to invest it.

  - He keeps stepping up the size of incremental ingredients to add, which incremental neighbors are happy to do because the underlying soup has ratched up its quality so investing in it incrementally more seems plausible.

  - The result is a delicious soup that everyone marvels is “just made from a stone”

    - Their own incremental additions all felt less important than the pre-existing soup when they were added to it, so they seem unimportant.

    - But the *totality* of the “garnishes” is large.

    - The soup gets its taste from all of the ingredients, not the stone.

  - AI isn’t too dissimilar.

  - The transformer architecture is a “simple” web of simulated neurons: a stone.

  - But then if you add to that stone soup all of culture and society, out pops an amazingly tasty dish.

  - But the reason the dish is so tasty is not the stone, it’s all of the other ingredients!

- LLMs didn’t train on images, but *pictures*.

  - An image could be any random bit of noise expressed as a 2D array of pixels.

  - A picture, in contrast, was a thing that a human *decided* to capture.

    - An intentful act of curation, an assertion that “this image is of a useful thing.”

  - Similarly, LLMs didn’t train on all plausible text, on text that *could* have been uttered; it trained on text that *was* uttered.

    - That some human at some point decided was useful to utter.

  - The strength of how common the pictures or utterances were in the training set was proportional to how useful humans, collectively, found it in the past.

  - LLMs generate text in response to whatever inane thing you ask them to do.

    - The human still decided to ask the LLM to generate the text, implying it is at least plausibly useful.

    - But what the LLM produces is always more “average” than what a real human would have said.

    - A small but consistent asymmetry.

  - That means that as LLMs generate more text, the signal of usefulness for text that exists in the world erodes just a little bit.

  - Fast forward many many years, and you get the heat death of the information universe.

- Imitation and innovation are both important.

  - Innovation is trying something new that is unlike what others are doing.

    - Innovation very often doesn’t work–at that point it’s not innovation, but just variance that turned out to not work.

    - Innovation is *expensive*. It takes effort and focus.

  - Imitation is replicating a thing you’ve observed others do that is likely to work.

    - If it weren’t likely to work, the other people you saw using it in similar contexts would stop doing it.

    - Imitation is what prevents humans from having to innovate all the time; once someone in society figures out a robust way to do something useful, we can simply do that in the future and take it for granted.

      - A kind of emergent caching of useful ideas.

  - In a system the force of imitation pulls us towards efficiency, the centroid.

    - In contrast, innovation expands the system’s horizons.

  - Without innovation pushing the horizons out, the system pulls to an extreme, rigid, brittle, centroid.

    - A heat death of the system.

  - LLMs just imitate, following the innovation that humans that came before have done.

  - In a world where we use LLMs for more things, how can we make sure that we keep the innovation in balance?

- Everyone is so focused on AGI that they’re ignoring applications of cheaper, lower end AI.

  - We’re so focused on pushing the upper edge of having higher and higher “IQ” of the top models.

  - We can now take an IQ of 90 for granted with cheap off the shelf LLMs… and it will only get cheaper.

  - What happens when you take IQ-90 LLMs for granted, and assume it’s too cheap to meter?

- Transformers have been such an explosive advance because they had all of the internet to train on.

  - All of that training data was pent up, ready to catch fire when the right spark came, giving a massive explosive boost.

    - We extrapolate from that boost to a runaway effect, but that might not be the case!

  - Now we’ve run out of fresh original materials, and it looks like the benefits from “simply scale the models more” is starting to peter out too.

  - We’re faced with a possible plateau where it will take lots of steady, patient progress to get considerably farther.

  - That means there’s a lot more time for society and the industry to catch up and figure out how to use this stuff!

- Someone told me about a tech talk from someone on TikTok’s data science team.

  - TikTok’s recommendation algorithm apparently works by creating a precise embedding of a given user, and embedding videos into that same embedding space.

    - This is a pretty typical modern recommender approach, I believe.

  - Apparently TikTok has found that the embedding is so predictive that if you select other users near a given user and look at their profile pictures, they *look* the same.

    - Same style of outfits, same basic facial shape, etc.

  - It kind of reminds me of studies about identical twins separated at birth.

    - They often share eerily similar behavioral quirks down to small details.

  - But if there are enough people in the world, the likelihood someone just like you--a non-twin "twin"--exists, just by happenstance.

    - A digital doppelganger.

    - And apparently TikTok's algorithm is powerful enough to find them.

- The web is more of a medium than a platform.

  - A platform is more about the programming model.

  - But what is more important about the web is its *distribution* model.

  - Web pages are like little shitty bits of software that would never be viable on their own, without the browser’s alternate distribution laws of physics.

- Game studios use Unity to coordinate the approximate actions of thousands of employees into a coherent game.

  - What would happen if you made an instance of Unity and opened it up to everyone in the world?

    - Would something coherent emerge, or would it just be noise?

  - Perhaps the thing that makes games built in this internal swarming approach coherent is a shared north star vision that everyone working on it is compelled to share.

    - If they didn’t share the north star they’d have never been hired, or would be fired.

  - Unity, at the end of the day, produces a coherent app: a point in time, coherent, convergent object that can be distributed in the app laws of physics.

  - Why does the output of Wikipedia cohere?

    - Everyone who chooses to participate, and who earns trust and authority in that community, adheres to north star principles that lead to a coherent outcome.

- Someone remarked to me that npm showed up precisely when it was necessary in the ecosystem.

  - But also, in some way, it couldn’t have shown up before the ecosystem wanted it.

  - There were presumably hundreds, perhaps thousands of little budding proto-npms that just weren’t in the right conditions to blossom.

  - At some point, the conditions were right and one of those lucky little buds blossomed.

  - It looked like the bud was the deciding factor (top-down), but in actuality the conditions were the deciding factor (bottom-up).

  - Similar to “the teacher will appear when the student is ready.”

- The games that overperform are the ones with a Just Right first hour.

  - Too much complexity thrown at the player feels like an overwhelming onslaught.

  - Too much heavy-handed hand-holding feels like an overbearing school, a chore.

  - A ramp that feels just right is engaging and keeps you staying with it.

    - Difficulty that scales right at the limit of your availability.

    - Hand-holding that is invisible.

    - Surfing the player’s flow state.

  - Product development for new kinds of products is more like a game onboarding than we typically think.

  - Ben Follington has an [<u>excellent new essay</u>](https://shimmeringvoid.substack.com/p/driving-and-dragging-loops) hitting on similar themes but with significantly more depth and nuance.

- One pattern to help people ramp up in a new system is to make a labyrinth feel like a maze.

  - A maze has many branching paths; navigating it is intimidating and challenging.

  - A labyrinth might be very intricate, but has effectively one path to go down.

  - Giving a player the *feeling* of it being a maze while actually being more of a labyrinth gives the best of both.

  - It feels like you’re doing something more novel and challenging than it is, giving a sense of accomplishment.

    - And because it’s one path, it’s easier for the creator to tune it carefully to be just the right difficulty.

    - It’s much harder to do that on a branching path!

- Finding the fun is like hunting for truffles.

  - Trust the person with the best nose for truffles.

  - That is, the best track record of finding the fun truffles in the past.

- If you're doing it for fun, and it's not fun, just don't do it!

- When people are having fun, they push the limits of the system.

  - They’re pushing themselves for its own sake, so they’re more likely to naturally hit the ceilings of the system.

  - When people are doing something because they’re told to, they’re more likely to just stay on the established paths.

  - If you’re trying to create an open-ended system, you want one where the initial users are pushing the limits.

    - If they aren’t, you might just be staying on safe paths shown off in the demos.

- Only when people are pulled forward of their own volition will they jump over obstacles.

  - If they’re being pushed, they’ll stumble or get stuck on even the smallest obstacles.

- My friend Matt Holden was talking about building a feature in his [<u>https://www.texipedia.com/</u>](https://www.texipedia.com/)

  - The feature is to allow a user to free-hand draw a symbol and then find Latex symbols that match.

  - The old way to do this would be an extremely complex, finely tuned system to do handwriting recognition.

    - The best open source implementation is some complex ball of Haskell.

  - The new way to do it… simply send the user’s scribble to GPT-4o.

    - The model is cheap and good enough to… just do it.

  - If you have a cheap oracle service that can answer your questions quickly and accurately, simply call out to the oracle.

  - And if you ever want to remove your dependency, *later* you can factor it out into a proper, locally-implemented approach in the Old Way.

  - But you might never need to!

- Networks used to have a permitered-based security model.

  - Everything within the castle wall was trusted fully; everything outside it was fully untrusted.

  - The most important thing was to verify that the castle wall was strong.

  - But of course, all kinds of dangerous things can get through walls.

    - Toxic slime can seep through tiny cracks.

  - The modern architecture is zero-trust network architecture.

  - You assume that nothing is trusted.

    - “Never trust, always verify.”

    - Every access request is authenticated and authorized regardless of source.

  - This allows fluid boundaries among everything.

  - By not trusting anything you can now get open-endedness.

- Novelty is difficult because it is a jungle of unknown unknowns.

  - When you trip over an unknown unknown you have a sidequest thrust upon you that you must resolve now… and that might not be *able* to be solved.

  - Unknown unknowns tend to branch and sprout off of one another into towering trees and jungles of fractal complexity.

  - Those jungles can absorb every ounce of your attention until the project runs out of runway and dies.

  - A common cause of death is getting lost in the jungle.

    - Side quests on top of side quests.

  - That’s one of the reasons to use boring, established, robust solutions wherever you can get away with it.

- A classic problem is knowing how many unknown unknowns you’ll run into.

  - The best bet is to use a boring solution that has been used in a large number of similar situations successfully.

    - The successful uses in the past wrung out the unknown unknowns.

  - But sometimes you have no choice but to try a solution that has some degree of novelty.

    - Perhaps for example it’s been used successfully in other contexts, but never in *this* organization.

  - The success or failure of novelty comes down to the rate of unknown unknowns.

    - How do you estimate the rate of unknown unknowns?

    - There’s no good way–that’s the fundamental problem.

  - The best way is to look at how many unknown-unknowns you’ve tripped over working on this project so far.

    - The unknown unknowns will all feel like random little niggling annoyances that required side quests.

      - They will feel like happenstances, but that’s the entire point–they were unforeseeable.

    - Look at the rate of how many of those have shown up so far on the journey.

      - Is it higher or lower than you would have expected?

      - How far into the journey to getting a viable solution does it feel like you? 20%? 80%?

    - If you’re sensing clear hints of viability, implying that you’re almost to the other side of the jungle, that’s a great sign that there’s only a small amount of remaining territory for unknown unknowns to lurk in.

- Your thing likely has much more novelty than you think it does.

  - For example, stitching together established components in a novel combination is in itself novel.

  - It might feel like the thing you’re applying a system to is 99% like what it’s been applied to before, but perhaps it’s more like 80%.

  - When you feel like you’re 80% of the way done, you’re likely only 20% of the way done.

    - The amount of novelty to navigate ramps up significantly in that last bit between the idea and deploying it successfully in the real world.

  - The benefit of using robust, “lindy” solutions is that the real world has wrung all of the unknown unknowns out of it.

- When faced with a too-large scope, it’s easy to get stuck bickering about which path is most expedient.

  - But if the scope is too big, if the novelty is too high, there’s *no* viable path through the jungle.

  - Bickering about which path to take will just waste your time.

  - The only answer is to massively reduce the scope to something that *is* viable, and also will get you closer to the other side of the jungle.

- Don't orient your system from the data layer up.

  - Infrastructure problems in a novel environment are charismatic traps.

    - You have to think about them to some degree, but they can absorb all of your capacity.

  - Orient it from product down.

    - What is the simplest thing that plausibly gives you the kind of product-level outputs you want?

- Momentum isn't good on its own.

  - Momentum is great when it’s pulling you towards a great outcome.

  - But momentum on its own is just a coherence in motion on the team.

  - The coherence can become a kind of rut.

  - Sometimes the coherence emerges organically in the absence of a viable vision.

  - Often the thing it coheres around is the charismatic traps: the kinds of puzzles that seem important and are fun to solve… and also are vast, dangerous jungles of compounding unknown unknowns.

- At each decision point, the easy thing is to shrug and keep going.

  - But now nine months later you're lost deep in the jungle.

- If you think you've got it, you'll keep going.

  - If you're stuck, you might be ready for a change.

  - If you're in a crisis now, use it to change.

  - The alternative is that you fall back into a dysfunctional homeostasis, a zombie shuffle forward.

- Imagine that you hacked through the jungle and made an elaborate treehouse on your own, Swiss Family Robinson style, including a working water wheel and rope bridges strung between the treetops.

  - That’s extremely impressive!

  - And yet also if an architect from the town comes to visit, don’t be offended if they aren’t willing to climb up to the top with you.

- Sometimes things are rickety in the details, and sometimes they’re rickety in the fundamentals.

  - It can be hard to distinguish the two cases at first.

    - If it’s your thing, you’ll think the ricketyness is in the details.

    - If it’s not your thing, you’ll assume the ricketyness is in the fundamentals.

  - If you can’t trust the fundamentals it’s hard to trust the overall thing.

  - The only test of which kind of ricketyness it is is if experts are willing to use it without being forced.

- Secret almond milk breeds disgust.

  - If you're handed something the person calls "milk" and it tastes funny you spit it out in disgust.

    - "This milk has gone bad!"

    - "No, it's *almond* milk."

    - "You should have said that at the start!"

  - The problem is not the almonds, the problem is the secret.

  - If you just own it then people's expectations will be more resilient to it tasting unlike normal milk.

- If you have a thing that will bite at some point, you have a choice to make.

  - Do you want to make it bite hard, at some indeterminate time in the future?

    - The bite, if it comes, will be hard, and possibly existentially dangerous.

    - But most of the time, you can simply not think about it.

  - Or do you want it to bite persistently but softly up front?

    - This makes it unlikely to cause unexpected problems down the line.

    - But it also means the user has to contend with a continuous, never-ending tax.

    - And also, maybe the user never would have gotten the hard bite in the first place?

  - The very first version of Google App Engine had a very odd development paradigm.

    - The framework and datastore felt wildly unlike the traditional LAMP stack at the time.

    - The benefit was that if you built your apps in this particular way, your app could scale to insane heights if necessary.

    - … But the vast majority of webapps that someone might build will never get more than a few dozen users anyway.

    - The tradeoff was to have a persistent, non-trivial tax, which made it less likely you’d ever bother to make anything worthwhile anyway.

- If your framework is “weird” in some specific way, that’s something that will bite the user at some point.

  - Users are more willing to put up with weirdness if they get some benefit.

    - “Do this slightly weird approach, but then you get (valuable bonus) for free!”

  - It can be tempting to hide the weirdness by layers of magic.

    - But then that might make it bite even harder when the user inevitably finds the edges of that abstraction.

  - The weirdness requires a leap of faith. It’s important for there to be a valuable landing spot on the other side to get people willing to jump.

- If your code doesn’t compile, is it the fault of your code or the compiler?

  - If it’s a compiler millions of people have used for years?

    - It’s definitely the fault of your code.

  - If it’s a compiler you just wrote and are using for the second time?

    - It’s definitely the fault of the compiler.

  - This is true for the weirdness of a library too.

- If you can’t reverse a decision before you die then it is effectively irreversible.

- Even if you’re playing a long game you have to survive the iterated short game.

- Most companies live in one pace layer.

  - If you cross multiple pace layers it's orders of magnitude more challenging, but also an order of magnitude more value that can be created.

- I find [<u>Ursus Wehrli’s images</u>](https://belopotosky.wordpress.com/2011/09/02/ursus-wehrli-organizes-the-world-pretty/) beautiful but grotesque.

  - He takes organic, emergent phenomena and then tidies them up.

    - For example, a sprig of a pine tree branch is decomposed into the individual straight twigs, sorted by length, and the individual needles laid out side by side.

  - A thing that occurs to me looking at it: the structure matters more than the details, but summary statistics don’t capture the structure.

  - This is one of the reasons that complex systems are so hard to grok.

  - Our reductionist tools are great for sorting similar things into buckets and counting.

  - But when you take a thing and put it into a bucket, you remove it from its context.

  - And [<u>context changes everything</u>](https://mitpress.mit.edu/9780262545662/context-changes-everything/).

- Having a stable, nurturing foundation allows you to innovate safely.

  - Alison Gopnik told me about a study of rats in a maze.

    - One arm of the maze had neither cheese nor a shock: low variance.

    - Another arm of the maze had either cheese or a shock: high variance.

    - Adult rats were more likely to take the safe but bland arm.

    - Young rats were more likely to take the high variance arm… but only if they could smell their mother near them.

  - The job of parents is to create a safe environment for experimentation.

  - You create an environment where it’s safe to “F around and find out”, by making the “find out” phase sting less.

  - A safe environment to fail allows much more innovation.

  - What if you could have a software ecosystem that capped downside significantly?

- Alison Gopnik has an interesting theory of [<u>empowerment</u>](https://www.socsci.uci.edu/newsevents/events/2024/2024-04-25-gopnik-1.php) as a bridge between Bayesian Causal Hypothesis Testing and Reinforcement Learning.

  - In reinforcement learning, empowerment is an intrinsic reward function for “maximizing the mutual information between \[the agent’s\] actions and their outcomes”.

  - This interpretation makes the claim that causal learning happens fundamentally due to empowerment and vice versa.

- A useful skill is to figure out who is the expert to trust in which domains.

  - Then when faced with a problem in a given domain, you can simply ask, “What would Dimitri do?”

    - The more trusted experts you know, the more domains you can have an effective oracle for.

  - You can induct who an expert is by careful study of who other successful experts defer to on what topics.

    - A kind of elo rating sorting.

  - If you are observant enough, you can derive a highly nuanced and high quality intuition about who to trust for what.

  - To do this requires:

    - 1\) Lots of time and patience.

    - 2\) The ability to pick up on subtle social cues.

    - 3\) The ability to discern where the edges of different types of expertise lie.

- You can’t frankenstein a formalism out of parts of other formalisms.

  - Formalisms are pure and tightly calibrated.

- Social complexity expands to fill all available space, up to the carrying capacity of the system.

  - This happens because social complexity emerges from a recursive process.

    - If I think about what you’re thinking, it gives me an edge to make a decision at an additional ply than you are, allowing me to best you.

    - But if you think about what I’m thinking you’re thinking, it gives *you* an edge.

  - There’s always an edge from thinking one additional ply than your fellow players.

    - Each player will individually be incentivized to add one more ply of thinking if they have room.

  - This is a runaway recursive process that absorbs all available space for everyone.

  - And now the system as a whole *depends* on this complexity; if you try to remove it, you might find that other emergent effects rely on it and that it’s load bearing.

    - For example, maybe you have an emergent ruinous empathy kind of culture; removing some of the plys of social complexity might move you to the obnoxious aggression quadrant… but with an employee population that has been selected for and honed for ruinous empathy, leading to an explosion.

  - The incremental step for an organization to gain more output capacity is to add an additional net head.

    - But instead of that net head going mainly to output, the new capacity goes mostly towards more load-bearing social complexity.

  - These emergent social phenomena emerge even if everyone is aware of the dynamics and dislikes them.

    - Political games are ones that no one wants to play and yet everyone is forced to play or be knocked out of the game.

  - An emergent, inescapable tragedy.

- The emergent politics in a large organization are kayfabe, which is kind of like LARPing.

  - LARPing is Live Action Role Play.

    - Historical reenactments are similar.

  - A key distinction: if you die in the LARP, you don’t die in real life… you just have to *pretend* to die.

    - If anyone saw you get hit, you’re obliged to pretend to be dead.

    - The more people who saw you get hit, the stronger the obligation.

    - But if no one saw you get hit, you can just… keep on going.

  - The kayfabe in an organization feels deadly real, but it’s more akin to a LARP than we often believe.

    - If you “die” in the organization, you’re still alive in the surrounding context.

    - This means that there are often clever moves you can do that look risky, but have capped downside.

  - Some people would rather have an “obnoxious aggression” culture than a “manipulative insincerity” to navigate.

    - I’m the opposite.

    - In a manipulative insincerity kind of environment, there are a lot more LARP moves.

    - People are unlikely to land visible “killing blows” in such an environment.

    - You can take advantage of that ambiguity to surf through and survive what should have been a game over.

    - The ambiguity is thick and swirling, but it also gives more space to deftly surf it.

- Lecan has a theory of the tension between neurosis and psychosis.

  - When you’re neurotic, you’re confused, stressed, and looking for answers.

    - In that state you become more closed.

    - But critically you know that there’s something wrong.

    - Therapy is highly effective in these situations to help you open back up.

  - When you’re psychotic, you have beliefs that are concrete and immutably clear: you don’t even realize you might be wrong.

    - You’re locked into a static worldview that might be completely incoherent with reality.

    - But crucially, you’re not even aware that you might be wrong.

    - This makes you very hard to coach in this environment.

      - You don’t even know to ask for or receive help.

    - Any coaching to bring you back to reality will agitate you further.

    - If you see anyone in this state, the best answer is to validate them to calm them down (if it won’t lead to harm) or to leave them alone.

  - The founder mindset promotes a psychotic process.

    - It stops you from being open, insecure, or questioning.

    - It forces you to believe you really are infallible.

    - Large organizations have an emergent kayfabe that the organization’s goals are infallible; the founder’s kayfabe is the belief that you individually are infallible.

  - Every founder believes they are right with every fiber of their being.

    - It’s the market that decides which subset of them is actually right.

- It’s exhausting debating with someone who is never willing to give you the benefit of the doubt.

  - If every single little detail or assertion has to be defended down to the ground truth, it takes forever to make even small amounts of progress.

  - At a certain point you just check out and say “screw this” unless it’s of existential importance.

- The understanding of something in your head is a hyperobject.

  - If you’re an expert, it’s a multi-dimensional hyperobject, perhaps of significant, fractal detail.

    - Note that some of those details will be real, and some will turn out to be illusory and not actually survive ground-truthing with reality.

  - But it’s impossible to communicate a hyperobject to others.

  - To communicate it its full detail would require you to exhaustively serialize its details, leading to a combinatorial explosion of detail.

  - That detail would take forever to actually unspool.

  - But more importantly, no one else will have the patience to listen and absorb it even if you did.

    - That would require an absurd amount of faith in the value of the hyperobject, because it could take *years* to absorb it all.

  - If you try to shortcut getting others to understand, you might get impatient at how they aren’t understanding something that seems so obvious to you.

    - That impatience (especially if you outrank the receiver) might make them even less willing to spend the time to understand.

    - This can become a toxic spiral of distrust.

  - To get the hyperobject actually absorbed, imperfectly, into others’ heads *requires* figuring out compressed requirements, concepts, and ideas that give significant understanding efficiently.

    - Perhaps it can get 80% of a given idea covered, in 20% of the overall complexity.

    - This is the leverage necessary to get even a part of the hyperobject into other people’s heads.

    - Finding the “backbones” of the ideas is an enormously expensive and challenging process to do.

    - And yet if you want others to understand and want to move in the same direction, it’s a requirement.

  - The only way to make progress in this kind of situation is to distill the simplest concrete smaller hyperobject that is on the path towards the longer term hyperobject.

    - That allows coordinating with others, and also starting the all-important process of ground-truthing.

- Just because you *want* to know doesn’t mean you *need* to know.

  - The time and effort to know–for others to help you know–can be extraordinary.

- Blind spots happen where you’re not curious.

  - You can be curious in lots of areas but differentially incurious in other areas.

  - This internal lack of curiosity in which domains you’re incurious in is a meta-blindspot.

- Living systems grow from seeds.

  - You can know the overall qualities it will have on a probabilistic level when it grows but never the details.

  - The details are totally sui generis.

  - The high level qualities are all that you can predict.

- When in doubt, just give it some space!

  - It's so simple.

  - It feels so counter productive.

  - "I need to *do* something."

  - But... do you?

  - Maybe it has agency all on its own, a desire to grow and develop, and just needs the space to do so.

  - Every time it grows stronger, it gets better able to tackle more problems on its own.

- We tend to find ourselves in recurring patterns of traumatic experience.

  - Perhaps it’s because the domain that we want to work in is inherently complex.

  - Or sometimes it’s something about our own behavior that causes the trauma to emerge again and again?

  - It’s always a mix of both.

- Humans narrativize *everything*.

  - If you put a handful of ideas in juxtaposition, readers will connect the dots into a narrative, automatically, without thinking.

  - You can use this to your advantage.

  - Put 9 dots into juxtaposition and let the reader complete the missing dot themselves.

- It’s possible to make ideas slow-viral.

  - Normal virality happens due to the aesthetics: surface level characteristics.

    - Sometimes it might spread further and faster than you want, especially if it’s a controversial idea in that context (e.g. the emperor has no clothes).

  - But it’s possible to be anti-viral on the surface level but super-viral on the fundamentals.

    - Only 9 out of 10 dots are connected; readers have to work at it for the insight to land, making it antiviral.

    - But when the reader connects that last dot, the idea is so compelling that they just *have* to share it, making it super-viral.

  - Caps downside, while keeping upside.

- Dandelion fields are only possible due to dead elephants enriching the soil.

  - [<u>Dandelion fields</u>](https://glazkov.com/dandelions-and-elephants/) are where new kinds of small-scale innovation shoots up.

  - Dandelions can only grow on solid ground.

    - The more unsolid the layer above, the more important the solidity of the layer below.

  - The “elephant” might be a pre-existing technical foundation, or your own prior elephant-style experience.

- The audacity of the vision is orthogonal to how egotistical the person is.

  - Apparently Einstein was quite humble.

  - He was just very focused on the ideas.

- You will under-recognize chores your spouse does.

  - The chore is only visible to you if the trigger for the chore is visible.

    - E.g. dirty dishes on the counter, or trash bins not out on the curb for pick up.

  - You don’t think about chores for their own sake; they are merely a means to an end.

  - No trigger, no thought.

  - When your partner does the chore, the trigger is gone.

  - There’s now nothing to trigger you thinking about it.

  - Which means you won’t think about it to thank them or recognize them for doing it.

  - In contrast, the chores *you* do will be hyper visible to you.

  - The result is you’ll think you’re doing more chores than your partner, and resent could creep in.

  - This is a built-in dynamic that can cause friction if you don’t acknowledge it and grab it by the horns.

- Neo was a part of the system of The Matrix, too.

  - Systems are more powerful than any particular person.

  - Even the people who are ‘Great Men’ are being driven along by the system.

  - The ‘Great Men’ are also chess pieces being moved about by cosmic systemic forces, just a different kind of chess piece than others.

- John Gall’s First-Law-of-Systems-Survival: A system that ignores feedback has already begun the process of terminal instability.

- My definition of worldly: someone who can have a truly interesting conversation with just about anyone.