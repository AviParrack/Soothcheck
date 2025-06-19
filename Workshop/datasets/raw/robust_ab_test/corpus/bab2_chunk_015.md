# 3/10/25

- [<u>Simon Willison notes</u>](https://simonwillison.net/2025/Mar/8/delaying-personalized-siri/) that the delay in enhanced Siri is likely due to privacy and security.

  - I agree that this is likely what’s happening.

  - It’s *extraordinarily* hard, in the current security models, to make something like this (access to all of your data as input, as well as dangerous tool use) safe.

  - I believe it requires an entirely new security model to do this well with any kind of scale.

- The current set of innovations around LLMs comes from harnessing Reinforcement Learning effectively.

  - (This section should be seen as my own intuitive distillation of phenomena that I’m not an expert in, I might get details wrong.)

  - Reinforcement Learning is remarkably effective at helping models “absorb” intuition in a particular speciality and get better and better at automatically doing well at tasks like what they trained on.

  - The problem is that RL takes an inexhaustible stream of ground-truthing, to constantly give the model feedback on what it should do to improve.

  - In some domains, you can get that ground truth feedback easily: for example, playing chess or go.

    - The “laws of physics” of chess games are easy to formally capture and verify.

    - You can have the model play other models and use normal computing to ground truth that the moves were all legal and which ones led to a win.

  - But lots of domains are hard to efficiently ground truth, for example they require doing an experiment in the physical world, or rely on human judgment.

    - A model during training waiting for human input on a given output would be like you asking the model a query and waiting a few millenia for an answer.

    - It would slow down the learning process by many, many orders of magnitude.

  - But once LLMs get enough generalized ability, they start being able to make high-quality judgment calls without a human in the loop for some tasks.

  - Instead of Reinforcement Learning with Human Feedback (RLHF) you get [<u>RL*AI*F</u>](https://arxiv.org/abs/2309.00267).

  - If you push too far beyond what the LLMs can do, you quickly get context collapse and the resulting models start poisoning themselves with slop input.

  - But if you keep it to areas the LLMs are extremely good at (e.g. domains where the LLM is so over-qualified it rarely makes a mistake) you can use RLAIF to derive a new model that is wildly better than before on that particular speciality orders of magnitude faster than if you had to have humans in the loop.

  - Now that LLMs are pretty good, a whole new class of specialities have become possible to do with RLAIF with high quality.

  - Theoretically it might be possible to use this technique to ratchet up model behavior by advancing iteratively on a number of domains.

- Claude 3.7 has clearly been specially trained to be good at generating React code and SVGs.

  - They are almost certainly using RLAIF to post-train the model to be significantly better in those domains.

  - It probably looks something like this:

    - Generate React code to accomplish the user’s request for a simple UI.

    - Then execute the code and poke at it with Playwright to get screenshots of it, and see how it responds to simulated user actions.

    - Ask an LLM to judge the quality of the result.

      - Does this look like what the user asked for?

      - Does it appear to work as a user would expect?

  - This gives an ability to get better and better at it by throwing a lot of compute at the problem.

  - Note that this only really works for React code (vs general programming) because frontend code is not algorithmically complex, but rather tedious.

  - LLMs do great at writing tedious code; they struggle with novel algorithmically hard code.

    - Put another way, LLMs can *use* React well, but it would struggle to write React itself.

- It’s possible for a model to over-fit to a specific framework, like React.

  - If I’m right that Anthropic has specially focused on React, I’d imagine the model got at least incrementally worse on non-React code.

  - The model likely “pulls” more heavily towards React, and that means it gets confused more often for things code in other frameworks.

  - Also note that this could mean the model won’t handle updates to frameworks well.

  - Tailwind apparently came out with a new, incompatible version that differs from the one that Claude 3.7 was heavily RLAIF’d on.

  - That means Claude 3.7 is actually quite *hard* to use with the latest Tailwind version.

  - RL improvements in models are at a low pace layer; it might take a year or more for the model to catch up to how Tailwind is now written.

  - There was always some hysteresis in developer ecosystems.

    - It takes a long time for sample code, documentation, and Stack Overflow answers to be written for the new framework version, and in that intermediate time everything’s a bit confusing.

  - But now with models specializing in particular framework versions, this hysteresis could get worse.

- [<u>John Collison on Twitter</u>](https://x.com/collision/status/1898907410187923919?s=51&t=vzxMKR4cS0gSwwdp_gsNCA):

  - “How does everyone manage state in the little personal apps they're vibe coding? If you believe AI software butlers will happen, there's a need for a product here. You want basic sync across desktop/mobile, good security/privacy, and (for extra credit) limited multiplayer.”

  - Someone should build such a thing!

- The main LLMs are improving fast enough that building an adjacent business is hard.

  - For example, a business to finetune models to specific use cases, or a carefully calibrated UX flow to help squeeze out higher quality React apps out of models.

  - But then a new model comes out that was directly trained on doing that use case well and it leapfrogs all of your linear improvements, obviating them.

  - “Proprietary tools to squeeze slightly better quality out of a model” doesn’t seem like a great business.

- Aficionados can distinguish small quality differences.

  - The same is true for coffee and LLM models.

  - They have enough experience with the various options and calibrated taste to be able to distinguish subtle differences and have informed preferences.

  - But the vast majority of people will think all of the options taste the same.

  - That means if your thing has a subtly better quality than the default option that can only be discerned by aficionados, you have a low ceiling; most people just won’t invest enough time to care.

- Malte Ubl has a [<u>take about about MCP</u>](https://x.com/cramforce/status/1898004300468830507) that seems directionally correct to me:

  - "OK, I'll say it, and I will age myself at the same time: MCP has J2EE vibes

  - Absolutely prepared to eats my words here"

- I loved [<u>Sam Schillace’s piece</u>](https://sundaylettersfromsam.substack.com/p/its-not-your-friend-its-an-api) about the problems that arise from viewing LLMs as humans instead of viewing them as a tool.

  - This is as far as I’m concerned the canonical take on the pitfalls of engineers using this incorrect mental model for LLMs.

- I loved [<u>What matters in the age of AI is taste</u>](https://sublimeinternet.substack.com/p/what-matters-in-the-age-of-ai-is).

  - Something I’ve been saying for awhile now, but canonically well distilled here.

- Ideas that people bring up in conversations are ones that they implicitly vote are interesting.

  - There are nearly infinite things that you *could* say in an open-ended conversation; the things you do choose to say are a tiny subset, the subset that you thought was *most* useful to say.

  - This is a consistent and significant bias in the kinds of things you say.

  - An idea that lots of people choose to talk about is intrinsically more likely to be interesting than a random idea.

    - Things that people take the time to write down in a personal letter are another threshold of quality.

    - Things that people take the time to write down in a book are another threshold of quality.

  - If you pull this curatorial judgment across all of humanity, something interesting comes out.

  - This is one of the reasons that LLMs can be powerful; they’re trained primarily on the most distilled collective votes for interestingness across society.

- LLMs are human-level reasoning at the speed of light, with infinite scalability.

  - A lot of things that require human-level reasoning have always had a friction that is governed by the "at the speed of a human" constraint.

    - That constraint is even stronger than it first appears, because you first have to find and engage a specific human in the task in the first place, which can be much harder.

  - This constraint forms a kind of force of gravity; omnipresent and unchanging.

  - But LLMs change that force of gravity, and suddenly a lot of things that everyone has just known are obviously impossible surprisingly become possible.

- A startup idea: "Mechanical Turk but the tasks use LLMs instead of humans"

  - Surely someone is building this?

- LLMs (including Deep Research) assume your question is coherent or a good one.

  - It’s very easy to accidentally trick yourself with some superficially good output on a fundamentally flawed question.

  - Model: "Turns out you were right all along!"

  - Human: "Just as I expected, thank you!"

- What makes the [<u>enchanted vault</u>](#k6vi4q1ipfp7) magic?

  - Data you put inside comes alive, helping you tackle meaningful things.

  - The more data you put in and the more intention you put in, the more it can help you.

  - The enchantment comes from collective intelligence; the combined indirect wisdom of everyone using their vaults around the world.

  - Even though everyone’s data is totally private; the overall system can still see the kinds of tasks that people are doing on their hidden data, and help automatically share those best practices, creating increasing leverage to everyone as more people use it.

- LLMs are kind of like caulk

  - That is, they are gap fillers

  - Good enough to fill in the details for you once you give the imprecise high level structure.

- Being precise is hard--you have to think through real-world fine-grained detail, and it takes time.

  - That's a high activation energy bar.

  - But LLMs can do a lot of that for you; you say an intention, it helps do the detail work for you quickly and with reasonable quality.

  - So the effective activation energy hump for humanity has gotten lower.

- When you're writing for other people, you have to meet your reader where they are as soon as you start affixing words down.

  - It's hard, and it front loads modelling how to reason about that topic with others.

  - But with LLMs, you can write your notes in whatever form makes most sense to you.

  - The LLM will be right there with you, understanding what you intend, adapt to you and be like "I got it!".

  - Then the LLM can help you distill that higher-fidelity hyperobject into more specific dimensionally-reduced versions for specific other audiences.

- Ecosystems that allow mechanistic emergence have to hit a certain critical mass.

  - A user’s use case is mechanistically met by the ecosystem if it can sift through everything that everyone has done before and find a good analogue to answer this user’s question.

  - These require significant critical mass to get going.

  - It only works for a given use case if either

    - 1\) some subset of the users are willing to do quite a bit of work, which can then be used for the rest of the ecosystem.

    - 2\) someone else that came before the user already had almost exactly the same problem.

  - LLMs lower this floor, because they are like a semantic lubricant.

  - They allow fuzzier matches to prior experiences to be viable.

  - Sometimes LLMs can give a good enough answer even for the very first user.

- Someone should write a manifesto that is optimistic about technology and also fundamentally human-centered.

  - So many perspectives today are either for tech as it exists today (centralized, extractive) or against technology (pessimistic about innovation).

  - These two things don’t have to be at odds.

  - It’s possible to have a vision for computing that is all of:

    - Optimistic about technology (including LLMs).

    - Centered around humans not corporations.

    - Cozy and human-scale.

    - Collaborative and prosocial.

    - Aligned with our collective aspirations.

  - The manifesto would be about unleashing tech’s potential to help humanity be at our best.

- I want soulful computing.

  - Technology that enables humanity to blossom into our collective potential.

  - Technology that nourishes humanity’s soul.

- I want the 80s PC DIY vibe but as safe and convenient as viewing a web page.

- The internet created the potential for a new kind of software.

  - It needed a catalyst.

  - The browser was that catalyst.

  - It was "just an application" in the old laws of physics, that is a portal to a whole new open-ended universe of software with otherworldly laws of physics.

  - The LLMs have again created a similar situation, creating the potential for something new.

  - You need a new kind of software as a catalyst.

  - "Just a web app" in the old laws of physics, that is a portal to a whole new open-ended universe of software with otherworldly laws of physics.

- Permissions prompts are kind of like responsibility laundering.

  - The system can't make a call itself so it asks the user a question they can't comprehend the implications of.

  - The permission prompt boils down to “do you trust this developer”, where “trust” is some ill-defined concept because it’s hard even for engineers to reason about what kinds of things saying “yes” might actually cause to happen.

    - You often can’t inspect the code even if you wanted to.

  - When the user says "yes" the system says "OK whatever, they said they were OK with it, so I guess the code can do whatever it wants.”

  - For example, if you say “yes” to a location permission prompt, the OS will happily allow the developer to send that location data to a marketing third party, even if that would be a surprise to the user.

  - There’s simply not enough granularity of the system into where the data goes and what it’s used for.

    - The only checks are at the boundary of the app before the information is passed to it.

  - Even if there were more granularity of oversight by the system, getting dozens of permissions prompts every minute would be overwhelming to users.

  - That’s part of why the ecosystem ended up with this equilibrium of coarseness of permission prompts within the security laws of physics we use today.

- What if you could atomize security models down to a finer level of detail.

  - Down to a point where users can make local decisions about local questions and also when it’s broken down to that level a lot of them are obvious.

  - It would also allow expressing policies with nuance that are inexpressible today. “Auto approve bank transfers to my spouse under \$200k”

  - There’s no leverage point to affix those rules in the system today.

  - Each individual use case / app doesn’t justify adding the complexity of that rule system.

    - Also before LLMs the activation energy was too low for users to actually use those geek mode tools.

  - If you move the security model for data flow not just within apps but across apps, then there's a central place to add this infrastructure and gate information flow.

  - In this central place a motivated user could affix rules of arbitrary nuance.

    - With LLMs even more users have the necessary motivation to do the precise tasks.

- The point of any security model is to allow users to accomplish their goals while keeping them out of trouble.

  - The effectiveness of the security model is how much people trust it, with minimal configuration, to not accidentally embarrass them or expose them to harm.

  - If the model feels error prone or not comprehensive enough, people will balk at ever collaborating with a thing that could leak stuff.

  - The model has to keep you out of trouble--help keep you from making a mistake that you didn't even realize was wrong until later.

- Letting untrusted code see your email is terrifying.

  - If it’s limited to a 1P service’s code that you allowed access to your email, it’s easier to trust.

    - You just need to trust that one entity to have good security hygiene and not be incentivized to sell your data.

  - But if it’s limited to code that 1P’s employees wrote, it falls into the tyranny of the marginal user; there is a low ceiling on the functionality that they can offer.

    - It’s either a one-size-fits-none feature (has to work for a lot of users, meaning it fits well for none), or they just don’t bother to build it.

  - To unlock the most value on your email you’d want an open ecosystem of 3P and LLM-written software.

  - But that’s inherently dangerous!

  - You’d need some kind of new security model to allow automated access of open-ended code from third parties while still being safe.

- Our emails are some of our most precious data streams.

  - There’s a mix of extremely useful, and also potentially embarrassing things in there.

  - Let’s imagine there’s some new feature that your friends tell you will change your life.

  - Which one would you prefer to get access to this feature?

    - 1\) Allowing some startup you’ve never heard of before to slurp your email to their servers.

    - 2\) Keep your email where it is, but randomly throughout the day without warning it shows some subset of your inbox to people who are physically nearby.

  - In some ways it feels like \#1 should be more scary–they could sell your data, or have lax security that allows hackers access to your financial accounts.

  - But the second one *feels* scarier, because the people who are near you are more likely to be people you’ll see again and again and now they might know something embarrassing about you.

- Why might a given piece of software that could be built not be built?

  - Sometimes it’s because it’s just not technically feasible.

  - But more often it’s because it’s just not economically feasible.

  - That is, it’s not worth the cost for someone to build and distribute it.

- Asymmetries create bias.

  - Bias creates alignment.

  - Alignment creates momentum.

  - Momentum creates outcomes.

- Trees, to reach the canopy, can't grow in all directions

  - They have to put their energy into growing in a particular direction: up.

- If you try to get consensus before you have momentum you'll never get momentum.

  - Momentum is all important; without it you can't actually have impact.

  - Consensus averages things.

  - If the momentum being averaged is not aligned, it averages to zero.

- Most interesting things happen on the edge.

  - The narrow dividing line between chaos and consensus.

  - Surfing that edge is where all of the potential lies.

- Is it noise or is it innovation?

  - Which anomalous data is worth paying attention to often comes down to a matter of taste.

  - If you can figure out in a general way which is which, then the universe would be your oyster.

- Consensus always pulls towards mush.

  - The centroid, the average.

  - Notably, that centroid might not itself be a viable point.

  - LLMs are inherently a kind of planetary-scale consensus mechanism.

    - They can give outputs that sometimes are at the centroid of a phenomena but not themselves part of the distribution.

  - For example, if you ask LLMs for “chicken paillard” recipes, they will do a good job.

    - The average of all chicken paillard recipes is a coherent centroid.

  - If you ask it to give you a “chili recipe” it is much more likely to give you a disgusting slop, asymptotically approaching vomit in appearance and taste.

  - Recipes that are published in cookbooks or even shared on TikTok had a real human in the loop asserting, “I tried this and it was good.”

  - The LLM can’t try the recipe itself, so it can serve you up something gross without realizing it.

- Consensus mechanisms don’t produce innovation.

  - Innovation is surprising, outside the distribution, at the edge between consensus and chaos.

  - Consensus mechanisms can only give innovative results if there’s a specific consistent bias in all of the components.

  - Imagine telling a room of creative people to individually come up with wacky ideas.

  - Then you take all of the ideas and average them together.

  - What you’d get is… the centroid, again!

  - Everyone was being creative, but they were doing it in random directions away from the center, so the average is still the center.

    - Also the average is likely to not even be a coherent or viable answer in the first place.

  - But now imagine you imparted a *consistent* bias to the creative process.

    - “Come up with creative ideas that build on a vibe from the Whole Earth Catalog applied to modern games.”

  - The bias is now consistent, which means that when you average it all out, you get something away from the centroid, and possibly itself innovative.

  - This intuition applies to brainstorming with a group of people, but also any time you’re using an LLM.

  - An LLM is effectively a planetary scale consensus mechanism, so it’s especially important to give them a specific bias in your prompt to get them to innovate.

- Models won’t push you to the edge.

  - The edge of the distribution is where innovation happens.

  - But models are consensus mechanisms that pull you towards the centroid by default.

  - You, the prompter, must give them a particular direction, a consistent bias that allows it to innovate in a direction away from the centroid.

- Once a project becomes auto-converging, everything changes.

  - Before that point, the project will fall apart if you remove the scaffolding.

  - If you don’t tell everyone on the team what part they should build, they won’t be able to figure it out.

  - Without some curation and “scaffolding” the activity of the team will just randomize; everyone’s best efforts will pull in random directions, pulling it apart.

  - Once it gets to the point where it’s clearly working and valuable, it becomes auto-catalyzing.

  - Past that point, the project has its own internal, auto-strengthening momentum.

  - It becomes obvious what incremental work to do to make the project better at what it already does.

  - At this point it’s free-standing, it’s alive.

  - It can stand under its own weight and grow.

  - At that point, even if you *tried* to diverge it it would be hard to; it has its own internal momentum and all the swarming energy around it from engineers and users gives it more momentum.

- Off-road teams are harder to get to do something coherent.

  - Part of the challenge going off road is there are no roads.

  - So you have no default schelling point to cohere around as a team.

  - Everyone pulls by default in a random direction, averaging to mush.

  - That means that as a team going off-road together, you have to have a clear northstar vision you all believe is important and are sighting off of.

- Everything everywhere for all time has been a remix.

  - We build on our priors, things we heard and absorbed in the past, and extend them in ways we vote are interesting or innovative.

  - Before computers and the internet, this process was illegible and hard to detect.

  - On the internet it can sometimes be extremely easy to see.

  - Our intellectual property schemes all assume total ownership over the work you made, as long as it’s sufficiently different from things others have copyrighted.

  - How much should your remix be worth?

    - The remix is composed of the underlying thing or things you built on, and the tweak you made.

  - First, assume that the ecosystem does find the remix valuable, and everyone wants to figure out who should get what proportion of the credit.

  - Conceptually the value of the tweak is tied to “if you hadn’t done that, how long would it take for someone else in the swarm of the ecosystem to make effectively the same tweak?”

    - If it’s “literally seconds later” then there isn’t much value that the creator should get credit for.

    - If it’s “thousands of years” then its’ extremely valuable.

    - Note that it’s not just the tweak, but deciding to build on that *particular* combination of inputs, out of the universe of all possible inputs, that is the innovation.

- Creativity is intrinsically inefficient.

  - Efficiency means “doing the status quo more cheaply and reliably”.

  - Creativity is variance outside the status quo, on the edge of chaos and consensus.

- I loved this old [<u>Hofstader article</u>](https://worrydream.com/refs/Hofstadter_2001_-_Analogy_as_the_Core_of_Cognition.pdf).

  - The process we use to think is also a constant building on top of ideas that have been useful so far, with a little variation.

  - We then keep the remixed things that turn out to be useful, and this compounds and builds on itself.

  - This is the process of “chunking” that allows significant and increasing leverage in the thoughts we can think.

  - The same process happens in society and within our minds.

  - The useful remixes are kept and built upon in a massive percolating, bottom-up, emergent sort.

- Every additional ply of thinking increases the difficulty by an order of magnitude.

  - The uncertainty you have to navigate compounds at each ply.

  - It gets harder to reason about.

  - But it also gets harder to execute, because the likelihood you missed something in your analysis also compounds.

- If you're able to accurately predict six steps ahead, that's not enough.

  - You also need to be able to survive while the world catches up those six steps.

- PMing of different types requires different plys of analysis.

  - Consumer PMIng - 1 ply

  - Platform PMing - 2 ply

  - Meta-platform PMing - 3 ply

  - Consumer PMing is insanely hard… so meta-platform PMing is practically impossible.

- There are positive and negative flavors of what is called “stickiness”.

  - Positive: the user gets more value the more they use the product and so they don’t want to leave it.

  - Negative: the user gets more and more stuck and can't leave the product.

  - Sometimes the two sides are related; as in a user storing more data in a system and building their workflows around it.

  - The more they store data, the more useful the tool gets for them… and also the more that it would be a pain to migrate everything out.

- Software is primarily social, not technical.

  - Like all social things, it must be grown, not built.

  - Engineering is a social activity.

  - Not individuals creating formally correct components separately that fit together.

  - It's a co-creative co-ownership.

  - Writing software together as a team is a social co-evolutionary process of generating understanding together.

- The schelling point for an ecosystem is often just as important as the protocol.

  - Git is the protocol; GitHub is the schelling point.

  - The schelling point is where people go for discovery of new things.

  - Without it, you don't know where the other good things that speak the protocol are.

  - GitHub and Git both need each other.

  - Git’s protocol design (with extremely lightweight branching) made a thing like GitHub’s forking possible.

  - GitHub’s ubiquity locked in git as the obvious protocol to use.

- Someone told me an interesting story about UXR from the earliest days of Excel.

  - They gave the tool to an accountant and watched how they used it.

  - The user put in all of the numbers in the table… and then got out their pocket calculator to do the sums.

  - They didn’t realize the superpower of computer spreadsheets.

  - The mental model was “paper spreadsheets, but in the computer.”

  - But by being in the computer, the spreadsheet could be magical and interactive.

  - A massively more powerful tool than the thing they superficially resembled.

- Kids today will never learn about filesystems.

  - In modern mobile OSes they're just totally hidden.

  - They've faded away, erased from history.

  - They're still there, you'd just never notice them if you didn't know they existed.

  - The filesystem is the thing that allowed escaping the same origin model.

  - And it's been erased from our collective memories!

- Infrastructure projects: infinite time gives logarithmic returns.

  - Quality projects: infinite time gives exponential returns.

  - The difference between mediocristan or extremistan.

  - The determining question is: is there a ceiling or not?

  - In the end there’s a ceiling for everything, but sometimes the ceiling is so far away that it might as well not exist.

- It can be hard in an ecosystem to get others to trust you if they don’t know you.

  - One approach is to go out of your way to make it extremely cheap for someone to detect if you cheat.

- Should software bend to humans or humans bend to software?

- No company wants to be the dumb pipes.

  - But everyone else wants the pipes they use to be dumb.

  - The model providers don’t want to be dumb pipes so they’re moving aggressively up the stack to the application layer.

- A micro-milestone on the path to PMF: needing to add a staging environment.

  - You need to add staging at the point where you have users who will be mad at you if something breaks on main, because they rely on the tool.

  - You have PMF for at least that one user!

- There are two main ways to get a team to move with coherence.

  - The first is to set a clear, compelling vision for everyone to sight off, that everyone is drawn towards.

  - Another approach that works when you have a lot of internally motivated people who are not yet aligned is to set constraints.

    - The constraints set a bias in the system; the random motion now has an asymmetry that pulls it in one direction.

- Coordination (within a team, or ecosystem) will take all the cost you're willing to give to it.

  - An insatiable social vortex.

  - "I thought we'd be doing most of the R&D on the novel architectural stuff, but we're just spending all of our time debating which Vite config to use."

  - There's power in schelling points that everyone can agree are reasonable and good enough and stop spending time debating.

- As a team navigating ambiguity it can feel like you’re going in circles.

  - But each time around to a place that looks superficially similar, you’re now individually and collectively wiser.

  - Seen from above it looks like circles; from the side it looks like a spiral, making progress in a third dimension.

  - As every team member absorbs the context and plays back their understanding, each accumulation of new insight, even if it’s mostly just repeating back what others on the team had already expressed, accumulates little bits of net new knowledge.

  - As everyone’s context is increasingly aligned, everyone’s different perspectives and insights can start finding the breakthrough insights.

- The likelihood of your workflow being broken by an upstream change in an ecosystem is tied to how many other people have a similar workflow and how loud they are.

  - There's safety in numbers.

  - If you're the only one with that workflow in the world, look out because the upstream might break it and unless you shout loudly enough it will likely stay broken.

- Everyone is high volition in *something*.

  - Some people are high volition in lots of things.

  - Some people are high volition in things other people find useful and are willing to pay for.

- All software rots, even well-written software.

  - The world changes, and so the software, fixed in time at the last time it was touched, no longer fits.

  - A given piece of software can, all else equal, be better or worse at resisting rotting, but it can never fully resist rotting.

  - The investment of maintenance energy counteracts rot.

- Building by addition and building by subtraction are fundamentally different.

  - Building up from clay or chipping down from marble.

  - Programmers tend to start with very specific, small things and then accumulate.

  - LLMs tend to start with a very general idea and then carve down into specifics.

- WYSIWYG systems almost always have some weird edge cases that are hard for users to reason about.

  - The fundamental reason is because the view is a reduced-dimension visual projection of the underlying semantic model.

  - In reducing dimensions, you must lose some of the nuance.

  - It’s possible for there to be two visually equivalent view states that are different semantic states.

  - That means when a user modifies the visual projection, the system sometimes has to make judgment calls about how to resolve the ambiguity in the underlying semantic model.

  - Often there are good enough rules of thumb that work as intended most of the time… but there are always possibilities for nasty surprises lurking.

  - If the system makes the wrong guess, the user might not even notice it for some time.

    - The projection of the incorrect state is the same as the projection for the correct state.

    - There’s no visual clue it got it wrong until later when the difference becomes obvious, but by then it’s confusing and harder to correct.

- The crystallized GUI is a distillation of institutional insight

  - But if you erode all of it it could become an overwhelming torrent of possibility.

  - Nothing is solid, there’s no terra firma.

  - Lost in a sea of infinite possibility, with no judgment calls of the people who went before you to guide you.

- If the reactivity is part of the magic of the system, then users won’t fully understand it if they aren’t able to see multiple views of that data update at once.

- When you have the wrong or ineffective mental model for a situation, it creates the possibility for a nasty surprise.

  - The wrong mental model that doesn't actually capture the relevant dynamics of the system gives you a faux confidence.

    - "I understand how it works, and it is doing this thing" when in reality it's not.

    - What Taleb calls the Turkey problem,

      - "My mental model is that the farmer is my friend who just wants me to eat well. It hasn't been shown to be wrong yet".

      - Then one day the incorrectness of your mental model is revealed to be disastrously wrong and you die.

  - Every mental model is wrong.

    - It must be. it’s projecting a multi-dimensional phenomena to a much smaller number of dimensions, which requires loss of information.

  - Yet some mental models are more wrong than others.

  - Ride the gradient of improving the effectiveness of your mental model, especially in high stakes situations.

- Knowledge grows from cycles of conjecture and criticism.

  - (Apparently this is an idea from David Deutsch.)

  - That is, form a mental model, a hypothesis.

  - Then expose that hypothesis to disconfirming evidence, for example ground truthing it.

  - Guess and check.

  - The disconfirming evidence gives you the information necessary to update your hypothesis to make it more accurate.

  - If you don’t make a guess, then there’s nothing to check, nothing to update.

  - Contexts where you can just passively absorb don’t require you to form a hypothesis that can be tested.

  - It’s easy to create environments that require active engagement in this loop:

    - Try to *do* the thing (instead of just reading about it).

    - Engage in a discussion with someone about the thing (being forced to actively distill your passively absorbed intuition).

    - Play a game that uses the concepts.

- I loved this video on [<u>Life and Entropy from NanoRooms</u>](https://www.youtube.com/watch?v=fzcVBRdI730).

  - Life as dissipative structures that create pockets of less entropy to better allow higher entropy creation elsewhere.

- Overheard: "How dare you exploit my laziness for your own personal profit!"

- A little piece of cozy software I want: a collaborative ELO ranking.

  - Put in a few dozen options, then show the user repeated comparisons between pairs of options and ask which they prefer.

  - Then calculate a ELO ranking of the options based on those preferences.

  - If other people can also vote, you could get an emergent team ranking on a question, and even be able to filter down and see individuals’ rankings, and compare where they differ.

  - This would be a useful little collaborative widget I’d use in various situations.

  - Any particular toy version of this would be hard to coordinate your team to use on some arbitrary platform, and there’s no obvious business model for someone to make one as a business.

  - A piece of software that I wish existed but doesn’t.

- It's not the steady state of being productive that's hard, it's the spin up.

  - If there are tasks that you find meaningful, then actually doing the tasks gives significant sustaining energy.

  - The harder the task is, and the less you care about it, the harder it is to get the energy to clear the activation energy hump.

- If you want something done, give it to a busy person.

  - When you're busy, you don't have time to say, "just one game of candy crush first",

  - You have to jump from one thing to the next with zero delay.

  - It's frenetic and stressful, but you also maintain the thread of hyperfocus and activity.

  - It's way easier to maintain that thread than to start it from scratch.

  - More tasks can fill in the gaps and help that busy person stay busy and productive.

- Large organizations can fall into a trap of being addicted to fire drills.

  - Everyone is aware that the organization has gotten slower to create value as it has scaled up, so everyone wants to show that *they* still have the hustle and aren’t the problem.

  - That leads to everyone looking busy, running around in circles doing no-leverage work.

  - Sometimes it’s not even no-leverage work but *anti*-leverage.

  - Destroying value by creating chaos that compels other people, who were previously doing useful things, to respond to.

  - This then compounds down the line; one person’s chaos creates chaos for others to respond to, which creates chaos for the next person.

  - If you say "I opt out of the performative fire drill so I can focus on doing the real high leverage work" people will think you're lazy, and possibly “the problem” of why everything is going slow.

  - That creates a very strong social pressure to participate.

  - An insatiable social vortex that spins faster and faster until nothing in the organization can escape.

- Embeddings are fuzzy but precise.

  - This allows them to be very good at capturing nuanced things that are hard to distill into formalized semantics.

- A pattern I use to make decisions in ambiguity: abduct a rubric out of my head.

  - Let’s say I need to choose between multiple options.

  - I introspect and try to figure out the dimensions that matter the most to me.

  - I then extract those dimensions into a column in a spreadsheet with either a continuous variable or a bucket (with a weight for each bucket).

  - I then try to introspect about my intuitive weighting of those factors and abduct a ranking function across those factors.

  - Then I put a bunch of items in, and see what ranking comes out.

  - If the ranking “feels” wrong I try to figure out what’s off.

    - Perhaps a missing dimension.

    - Perhaps the weighting of a factor is wrong.

  - Then, I just keep doing this iteratively, putting in more options and tweaking the rubric until I have a final choice I feel good about.

  - The rubric doesn’t tell me how to think, it helps me sharpen and engage with my intuition.

  - I wish I had tools that would help me do this process!

- People intuitively want to avoid learning about complexity because they fear it will slow them down.

  - But it will actually speed you up to learn about gravity, instead of constantly running up against it.

  - Now you can focus your time and effort on things that might work, not wasting time on things that definitely don't.

- If you think you are extremely rational and not prone to emotional manipulation then you are extremely emotionally manipulatable.

  - We can’t see things in our blindspots.

  - If there’s a core part of us that’s in our blindspots, when we are manipulated there we won’t see it.

  - If there are important and high leverage parts of the system, even if they are parts you’d rather not admit are important—*especially* if they are parts you’d rather not admit are important—you’d better admit they’re important and plan around it.

  - Ignore a force of gravity at your own peril… especially if it’s one that others can steer to their own ends.

- Try to use the MGI: Most Generous Interpretation.

  - A good faith interpretation.

  - Helps build bridges of understanding.

  - If you don't use MGI, you're using others’ actions to help achieve your emotional goals and needs

    - "See, they don't *want* my help."

  - Why do people not do this as often as they should?

    - It’s related to the Fundamental Attribution Error.

    - The system that constrains us is inescapably obvious to us, but hidden to everyone watching us.

  - Curiosity is the antidote.

  - “Assuming this person is collaborative and competent, why did this bad outcome still happen?”

  - You’ll assign less blame, allowing you to learn more.

- My personal serendipity engine works by cultivating intellectually meaningful relationships with people I run into who are interest*ing* and interest*ed*.

  - Being an extrovert means I at least enjoy almost every conversation I have with any random person, capping the downside.

  - But each conversation I have with an interesting and interested person, someone who wants to talk and explore ideas together, has the upside of possibly discovering a game-changing idea together, now or in the future.

  - Capped downside, uncapped upside.