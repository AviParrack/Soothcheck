# 4/14/25

A [<u>recorded conversation with Aishwarya Khanduja about this week’s reflections</u>](https://share.descript.com/view/BCOaHzmnoUb). [<u>Previous episodes</u>](https://common.tools/common-ground).

- [<u>Most Americans don’t trust AI or the creators of AI</u>](https://www.theverge.com/ai-artificial-intelligence/644853/pew-gallup-data-americans-dont-trust-ai).

  - There's an opportunity for someone to paint an optimistic, human-centered vision of AI.

  - A vision that isn’t simply “let’s do more of the same with tech” but that grapples with tech’s indirect effects on society.

  - Technology, properly situated in society, can be a massive, optimistic unlock.

- I thought this was a great point in Venkatesh Rao’s [<u>Terms of Centaur Service</u>](https://contraptions.venkateshrao.com/p/terms-of-centaur-service):

  - "My AI expanding my one-liner prompt into a 1000-word essay that is summarized to a one-liner tldr by your AI is a value-adding process because your consumption context is different from my production content."

  - There’s the joke about how in the age of AI everyone will write bullets that are converted into a formal email, and then the reader will take the formal email and reduce it back to bullets.

  - But that joke misses something important: the expansion and distillation process can be *specific to that person*.

    - For example, if I’m familiar with the concept of complex adaptive systems, a summary of an email might be able to elide that concept where for other readers it would have to be included.

  - So that means there is information being added in the expansion and distillation steps.

  - This is the power of liquid media.

  - Media that can be adapted to each listener and give them what they need to know.

- MCP could help interact with complex pro tools more naturally.

  - I thought [<u>this example of using MCP to control Blender</u>](https://x.com/youwillmakemaps/status/1908228072962183467) was interesting.

  - Blender is an extremely powerful application, but also famously hard to use.

  - But MCP instrumentation allows interacting with Blender using natural language.

- The more high-quality context the system has the less work the user has to put into the prompt to get a good result.

- We need a new digital home for the age of AI.

  - A place to keep the memories for our AI life.

  - A central vault for everything that’s important to us.

  - There’s always been the dream of a universal account for our personal data, but now with the power and promise and danger of AI, it becomes an existential requirement.

  - Your digital home must be something you own, not rent.

  - It must be fully your turf, no one else’s, and fully under your control.

  - It would need a new privacy model for third parties to do useful things with data without leaking information.

  - It would need to be an open, decentralized system so it could be ubiquitous.

- I want LLM pixie dust infused into normal UIs.

  - For example, An LLM-assisted todo list.

  - What if a given UI could use that pixie dust to change itself to better fit your needs?

  - Not just malleable, but *adaptable*.

- I resonated with [<u>this argument about focusing on agent’s reliability vs capability</u>](https://www.sergey.fyi/articles/reliability-vs-capability).

  - The gee whiz demos are ones of *capability*, e.g. “book me a flight.”

  - But the user value of that capability is highly dependent on its *reliability*.

  - If the automation fails, it often takes *more* time than it would have if you hadn’t used it in the first place.

    - You invested time to configure and execute the automation.

    - When it fails, you have some amount of time and effort to diagnose what went wrong and what you would need to do to fix or unwind it.

    - You now need to do the task manually anyway.

  - Let’s analyze a hypothetical use case.

    - The use case takes 10 minutes to do manually.

    - If the automation works, it takes 5 minutes.

    - If the automation fails, the whole use case takes 20 minutes.

      - 5 minutes to execute the automation.

      - 5 minutes to diagnose the problem.

      - 10 minutes to do the task manually.

    - The automation has a 60% success rate.

    - The expected time of using the automation is 11 minutes.

      - This is longer than the 10 minutes to just do it yourself.

      - The automation is under water.

      - Over time, as more people try it and fail, and update their priors for the success rate (seeing how successful it was for them in the past, or for their friends or other users), over time the expected use of the underwater automation is 0.

  - The three terms that can vary are:

    - What percentage of task time is saved if the automation works?

    - What percentage of task time is lost if the automation fails?

    - What is the success rate?

  - The gee whiz use cases tend to actually be underwater: there are a lot of steps, all of which must work correctly in sequence, for the automation to fully work.

  - The simple, dependable cases are often viable, and from there you can grow into more and more complex scenarios as the system improves.

- Tools like Loveable allow vibe prompting to create apps.

  - But the thing that’s created by them are ultimately normal webapps.

  - Those complex apps can be difficult to administer if you haven’t written webapps before.

  - As it escapes the creator’s ability to understand it, it gets increasingly unwieldy to maintain, augment, etc.

  - If you ever have to think about npm scripts, you’re no longer just vibe prompting, you’re vibe coding.

    - This sets a ceiling on who can do it.

  - Users want a substrate where the actual ‘app’ is extremely simple because the framework gives it all of the data, integrations, etc it needs.

- LLMs are extremely confusable deputies.

  - In security, one type of vulnerability is the [<u>confused deputy</u>](https://en.wikipedia.org/wiki/Confused_deputy_problem).

    - A powerful entity is tricked into applying their powers in a way the user didn’t intend.

  - LLMs are inherently gullible and extremely confusable.

  - That means you can’t give LLMs that have been provided untrusted input any kind of power.

  - That’s the core of the prompt injection problem.

- Prompt injection is the fundamental problem to address to unlock the power and scale of AI.

  - Without solving prompt injection you can either get power or scale from AI, but not both.

  - This [<u>overview of MCP’s prompt injection problem</u>](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/) from Simon is great.

  - This [<u>Camel technique</u>](https://simonwillison.net/2025/Apr/11/camel/) is an interesting one, as profiled by Simon.

    - It’s a more limited and specific version of a [<u>solution sketched out a couple of years ago</u>](https://www.wildbuilt.world/p/safer-ai-agents-with-ifc) by Berni Seefeld.

- Adding the internet radically changes the security model of a system.

  - In some cases it's just fundamentally impossible to retrofit.

  - Windows 95 was never made safe because it was designed in a world before the internet and that threat model.

  - It was a dead end that had to be routed around with Windows NT.

  - MCP fundamentally grew out of a local, trusted environment; it might be impossible to retrofit the internet onto it.

- Prompt injection is actually a specific case of a more general problem that has been around for decades.

  - That general problem, believe it or not, is the cause of hyper-centralization and the dominance of one-size-fits-none software that is hyper optimized to engagement hack us.

  - We take it for granted that it must work this way because the laws of physics set our horizon of what we can imagine.

  - But the same origin paradigm is not preordained.

  - It is simply one among many models we’ve used, up to its maximal point.

  - But it can’t take us where we need to go.

  - The promises of LLMs are too great, straining this model.

  - We need a new model that can take us beyond.

- Personal context would improve the behavior of LLM-based systems, but is fundamentally risky.

  - There have been attempts, like [<u>RFC 9396</u>](https://www.rfc-editor.org/rfc/rfc9396.html), to describe how more fine-grained information could be permitted.

    - For example, you could express things like “only expose information that matches this regular expression, and is no more than 7 days old.”

    - But those limitations are hard to administer, and still too binary and black and white.

  - For example, I’d be OK with a system that generates an insurance quote that can look at a wide swathe of my information–as long as the *only* thing the insurance company could ever learn directly is whether I’m approved or not at the end.

    - The insurance company would also want confidence that their algorithm was faithfully executed on real data, even if they can’t see the data.

- [<u>An intriguing argument</u>](https://julian.digital/2025/03/27/the-case-against-conversational-interfaces/) about the role conversational interfaces might play in our UIs.

  - “The inconvenience and inferior data transfer speeds of conversational interfaces make them an unlikely replacement for existing computing paradigms – but what if they complement them?”

- In a world of agents there will be an arms race between users and providers.

  - An example of this arms race heating up: [<u>Cloudflare has a new service where it feeds plausible believable slop to suspected bots</u>](https://www.theverge.com/news/634345/cloudflare-ai-labyrinth-web-scraping-bots-training-data).

    - This wastes the bots time and makes it not worth their while.

  - This kind of weaponized chaff will likely get worse over time.

  - Screen scraping gamesmanship has been a thing for decades, with agents it will get even weirder.

  - For example, I’ve heard DoorDash is trying to block agents from using its service.

  - Many services like Doordash have a business model that presumes not a dumb pipe of functionality but an aggregator model.

  - They collect all of the user demand, and then can use that powerful position to get leverage over the supply–their ability to steer to different providers is powerful.

  - But part of that steering ability is because humans are easy to distract.

  - When they see an ad in the corner of their eye or a tantalizing upsell, they just might click it.

  - Bots are way less likely to click, which means that the business model of the aggregator reduces to more of a dumb pipe: a much worse business.

- LLMs can make generalists almost as good as specialists in many domains.

  - The generalist meta-skills of volition, savviness, curiosity are now more important than the expertise.

- Vibes aren't everything, but in the absence of strong fundamentals, they at least give you the benefit of the doubt.

  - Strong fundamentals means that the thing is resonant and rigorous--the closer people look, the more convinced they become.

  - Superficially-messy-but-strong-fundamentals (beautiful mess) is better than shiny-but-poor-fundamentals (gilded turd), but when it's superficially-messy-and-weak-fundamentals it's the worst of both worlds.

- The vibe of [<u>A2A’s documentation</u>](https://google.github.io/A2A/#/topics/a2a_and_mcp) is much worse than MCP’s

  - The documentation for A2A looks hideous and very "old school engineers who love java" vibes.

  - Compared to MCP's "savvy developers who use all of the modern tools and have a sense of style" vibes.

- Claude 3.7 is like a chainsaw, it over-extends what you asked it to do.

  - Even if you ask it to change just one file in your project, it’s likely to remodel your whole codebase.

- Kevin Roose: LLMs are the world’s most insecure intern.

  - Constantly asking for your permission, messing stuff up.

  - I’d also add “overzealous”.

- Platform features that require coordination often can’t be “solved” in userland.

  - Userland is the region of the platform where platform users can do whatever they want on top of what the platform creators made.

    - It’s easier to change, and thus is a higher pace layer.

  - Subsuming functionality into the platform is expensive and should only be done when there’s some benefit to doing it in the platform.

    - An overly-large platform gets harder to maintain and reason about, for one thing.

  - However, there is a class of problems that can be solved *technically* in userland, but are actually a coordination problem.

  - For example, many years ago Javascript had no notion of Classes.

    - You could create class-like things in userland by swizzling prototypes, and lots of different libraries had subtly different conventions.

  - This led to less interoperability than was ideal.

    - No schelling point could emerge; there was a sea of subtly incompatible options.

    - If you used an object from another framework it might have a subtly different lifecycle for no good reason.

  - Then Javascript formally ensconced one of the notions of Classes into the language.

    - It just added “sugar” for one of the conventions.

    - It subsumed that one convention down into the platform layer.

  - This instantly solved the schelling point problem in userland.

    - Now there was no reason not to simply use the one official way.

  - Adopting one convention in the lower pace layer solves the schelling point problem immediately.

- When you're speeding down the runway, the moment the wheels lift off the ground, nothing feels like it changed that much.

  - But it's an infinite difference.

- I like the metaphor of [<u>sleepwalking geniuses</u>](https://zicklag.katharos.group/blog/im-tired-of-talking-to-sleepwalking-geniuses/) for LLMs.

  - It captures how powerful they are… and also how silly they can be if you don’t constantly guide them.

- The eval loop is the beating heart of a quality improvement process.

  - For example, a search quality problem.

  - The loop: sample sessions that had a bad result, come up with scalable ways of improving them, experiment, ship, repeat.

  - At the very beginning it’s hard to get that eval loop humming.

    - You have to make it turn a few times before it gets going under its own steam.

    - Kind of like hand-crank starting a car motor.

  - The eval loop can absorb all of the attention and resources you give it.

    - It can absorb infinite energy.

  - So be careful to only give it the proper amount of attention.

  - If the core product has PMF and you’re still getting super-linear returns from the loop, then keep investing more in it.

  - But if the eval loop is for a secondary part of the product, or a product that doesn’t yet have PMF, or is getting significantly diminishing returns because you’re hitting the quality asymptote, pull back resources.

- Remote is way harder than in-person for brainstorming.

  - One reason in-person works is that when one person gets spun up the other person can meet and sustain that energy without it dropping.

    - There’s no delay, no asynchronicity.

  - Also, over VC only one person can talk , so everyone has to think “is this worth wasting everyone’s time?"

    - A much larger chilling effect for introverts than extroverts.

- Trusted Execution Environments (TEE; AKA Confidential Compute) seem to be getting more interest in crypto circles in the last year.

  - A much-easier-to-deploy technology that provides a lot of the benefits of ZK Proofs.

- A weird game-theoretic equilibrium in organizations: schedule chicken.

  - Five teams are asked "are you ready to launch" and they all say yes, even though none of them are.

  - They all assume that another team will cave first and say they aren’t ready, and get egg on their face, saving your team’s reputation.

  - The boss erroneously thinks the project is ready to ship: kayfabe.

  - But sometimes, especially if there’s little psychological safety, no one caves, and so a ruinously unready product ships.

- Corporate politics expand to take all available space over time.

  - Corporate politics don’t arise because of human foibles; they arise due to the game theoretic edge that an entity gets from playing incrementally more politics than their peers.

  - Everyone gets an edge, so everyone is incentivized to play politics.

  - That then sets the baseline amount of politics up, and the cycle repeats, and continues compounding.

  - If you don’t play as much politics as your peers, you’re likely to be knocked out of the game by more political players.

  - The only thing that contains the expansion of politics is the ground truthed success of the company in the wild.

  - Imagine if you had a swarm of LLM-backed agents in a company.

  - They could conceivably absorb infinite energy, all being consumed playing a self-referential loop of corporate politics!

- The measure of interestingness is the ability to surprise.

  - I believe this idea comes from Eliezer Yudkowsky

  - Zealots are unsurprising.

  - The topic they are a zealot about is infinitely important to them, and dominates all of the other topics.

  - That lack of surprise overshadows all of the other possibly surprising things they might say.

- Humans sometimes surprise you, but algorithms mostly just confirm what you asked for.

- "Don't get it original, get it right."

  - I believe this comes from Edward Tufte.

  - If there's an existing pattern that works elsewhere and isn’t your differentiator, just use it!

- Heart transplants are often not a bad idea.

  - However it is always a bad idea to do a heart transplant with a butter knife.

- Scale hollows everything out.

  - At a small scale you don’t think about the nice touches cost, you just do them.

    - The direct cost might be high; the indirect benefit might still make it worthwhile.

  - For example, if you have an individual home you rent on Airbnb, you of course stock the kitchen with the basics.

    - You don’t even think about it.

    - It’s below the Coasian Floor.

    - Of course you do it!

  - But say that you’re a multinational corporation with hundreds of vacation condos you own and rent out.

    - How much does it cost to have vegetable oil in each of a thousand units?

      - At scale, the answer to even a small expense is “quite a lot!”

    - Now it’s above the Coasian Floor.

    - You *must* think about it.

    - The question you ask yourself is: "would not having this make customers noticeably less likely to choose to book again in the future?"

  - The costs are legible; the benefits are illegible, so the cost cutting dominates.

  - Each individual nice touch clearly isn't load bearing on its own, so you cut it from the budget.

  - But together, *all* of the little touches were load bearing, making the rental feel more soulless, not the kind of a thing the user might want to return to.

- Going for more scale incrementally makes sense, but gets you stuck in local maxima.

  - Going for less scale than today almost never does.

  - Companies walk up the scale ladder until they're trapped with a local maxima.

  - The good enough product for the largest market they could reach from their starting point.

  - But now they are stuck, and have nowhere to go from there.

  - You can only adapt when you change, and if you can’t change without going down the hill, you don’t adapt.

  - Without adaptation, at some point the environment changes and you die, stranded on your local maxima mountain.

- Not thinking through the implications of your actions is a form of externalized-risk leverage.

  - Like all leverage it allows you to go fast by taking on more danger.

  - But normal leverage only puts *you* in danger.

  - Whereas not thinking through the implications puts everyone affected by the negative second-order implications of your actions in danger.

  - For powerful entities, the indirect effects might *primarily* impact others.

  - You think you’re going fast, but actually you’re doing it primarily by externalizing the risk to others.

- I want an enlightened approach to technology.

  - A worldview that embraces more than just Computer Science as a lens.

  - A worldview where creators grapple with the indirect effects of their actions, and optimize for a net positive impact on society.