# 9/9/24

- Things written by LLMs are slop.

  - Integration code written to combine disparate things is glue code.

  - Glue code written by LLMs is **glop**.

  - Glop is a kind of black box; it doesn’t need to be understood necessarily.

  - “The integration broke!” / “Just shovel in some more glop until it works!”

  - A large proportion of code in the world is glue code.

  - In the future the vast majority will be glop.

- LLMs can be used as compilers.

  - They compile English to code.

  - This is extraordinary!

  - As a creator, you can sketch out real code and English intermixed.

  - The LLM can compile it all, bake it into plain old code that doesn’t require an LLM to execute.

  - Once it’s compiled to plain old code, it can be run cheaply.

    - You want the LLM to only be called when there’s something to change, not when you run it.

  - This section is a riff on [<u>this tweet</u>](https://x.com/jamespotterdev/status/1830181400408133893?s=46&t=vBNSE90PNe9EWyCn-1hcLQ).

- The most important thing to drive LLMs is to curate good context.

  - With the right context, LLMs are *very* good at producing high-quality output.

  - The hard part is no longer the magical thing working on your data, it’s having the relevant data all in one place.

  - Today’s chatbot models for LLMs give an implied context: the previous parts of the conversation.

    - This allows you to accumulate more context as you converse with it.

  - But this is a limited form of context curation, because it is implicit, and more importantly, append only.

  - Once you find the right context in a conversation, you want to be able to prune the unrelated parts, the meandering dead ends from the conversation that might confuse the LLM in the future.

  - The more tightly tuned the context, the better the answer.

  - You can think of this curation as gardening of contexts.

  - How can you design a product so that incremental gardening feels natural, easy, and also productive?

  - In such a system, the user would be doing creative work, steering the LLM, but it would feel like a byproduct of doing gardening work that already made sense to do on their data.

- People who can code via LLMs don’t necessarily have an intuition for what plain old code can do.

  - That leads to, for example, trying to create an Anthropic Artifact to identify what kind of dog is in the picture.

  - But Artifacts can’t call out to LLMs when they are executed.

  - They are compiled from english to code, but once that’s happened, they’re just plain old code.

  - Claude will happily write such an artifact, with a comment saying “call out to a proper image identification service here”, but to the non-technical creator, they don’t realize that, and just [<u>think that Claude isn’t very good at identifying dog breeds</u>](https://x.com/yescynfria/status/1831009523991224778).

  - It's confusing that you can use LLMs (magic) to write an Artifact (plain old code) but only at write time, not at use time can you use magic.

- Artifacts were a low-hanging fruit made possible by LLMs, just waiting to be discovered.

  - How Anthropic Built Artifacts: [<u>https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts</u>](https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts)

    - The feature went from initial demo to production launch in 3 months.

  - Now that they exist, it’s obvious there’s a there there, and that using LLMs to whip up some glop is a key use case.

  - But Artifacts as they exist have a lot of room for improvement.

    - They can’t use LLMs themselves.

    - They can’t compose Artifacts to build more complex functionality.

    - They can’t safely use or store a user’s data.

- Financial data and medical data (data from doctors, not fitness data) never got aggregated by consumer aggregators.

  - These are data sources that could have all kinds of low-hanging fruit use cases, if only there were a place that users felt in control to safely manipulate their data.

- A useful characteristic of a malleable system: "Squish to fit.”

  - Allow imprecise / jury rigged connections, allowing experimentation without being too precious about it.

  - That was extremely hard to do automatically before LLMs.

    - You had to significantly cut down on the things that could be done, to the subset that could be made tinkerable.

  - But LLMs are squishy!

    - They can squish things!

    - Magical duct tape.

- When you center data, not code, it becomes easier to design malleable systems.

  - Most systems that use code center the code, not the data.

  - But it’s possible to center the data, and have the code be a bolt-on after thought.

  - Little ofudas, spells attached to bits of data.

    - Ofudas are a hybrid of prompting and scripting.

    - They can be compiled by an LLM down to just plain old code in many cases.

    - Ofudas enchant an object.

  - This pattern is not new.

    - In Dynamicland, objects broadcast what actions they can perform.

    - In the Sims, individual objects broadcast what actions they can help with.

    - In many game engines, it’s possible to attach little Lua scripts to game objects.

  - Each individual enchanted object is pretty ordinary; it’s the emergent outcome of multiple enchanted objects interacting that creates the deep magic.

- When designing a malleable system, what kinds of components do you need?

  - There are a small number of extremely common, moderately complex components that are horizontal, finicky to create, and general purpose.

  - A candidate list:

    - Table View

    - Image Collection

    - Rich Text Editor

    - Code Editor

    - Calendar

    - Indentable Outline

    - Map

  - These are the foundations of the UI that tinkerers can glue and wire together.

  - You want to provide these as built-ins that everyone can take for granted, and that can be created to a high polish.

  - Everything else is glue and glop.

  - Glop is rarely polished, but if the components that are the majority of the UI are polished, and the glop just connects and fills in data, then the result can still be reasonably polished.

- James Cham reacting to last week’s Bits and Bobs: “Merely good software that is exactly what I need is going to be great.”

- A year ago: "Who could ever need more than 200,000 tokens???"

  - And yet today every day I run into context / conversation length limits in my Claude chats that use projects that I’ve crammed full of background context.

  - LLMs do better the more context they have to work with.

  - People will cram contexts as full as the LLM will let them.

- The Zombo.com problem: you can do anything in theory and nothing in practice.

- If it never changes it’s invisible even if it’s huge.

  - We only see things that change, if they don’t they fade into the background noise, we don’t recognize them.

  - There’s tons of friction that is invisible to us because we don’t even realize it *could* be different.

    - “:shrug: That’s just the way that is.”

  - But if the laws of physics change, sometimes friction that used to be a fact of life turns out to be possible to bend… or even break.

- "Out of sight, out of mind"

  - We notice the presence of something many many orders of magnitude more strongly than the absence.

  - The absence is easy to forget about.

    - You have to maintain a mental model of the now-hidden thing, and keep refreshing that memory or it evaporates from your awareness.

  - The presence is much harder to forget about--it's right there, you can see it with your own eyes!

    - Each time your eyes pass over it it reminds you it exists.

  - One of the ways we think larger thoughts than our mind can handle is by caching partial computations in the world around us, ready to be pulled right back in in a moment by our eyes.

    - A form of stigmergy.

    - Our working memory is like the register file.

    - Our memory is like RAM.

    - The objects we create in the world (e.g. things we write down) are storage on disk.

- For any given piece of crap, there's a demo that makes it look pretty good.

  - One of the reasons you should use the system you're building for real and not just demos.

  - If you only demo, you'll be drawn to the parts of it that demo well and work, and away from the parts that won't.

  - You'll erroneously conclude "this kind of works," but it actually doesn't.

  - You're just focusing on the parts that *look* like they work.

  - As you avoid the parts that don’t work, those parts become increasingly in your blind spot.

    - They become invisible to you, even if they’re huge.

  - Whereas if you use it for real, you focus on the parts that you need to use, whether they work or not.

  - When you try to interact with a part that doesn't work, and are in a position to fix it, the system gets better and more capable the more you use and improve it.

- An hour-glass shape of the system gives both ubiquity and variety.

  - Typically the more options you have at a given layer, the more challenging it is to evolve the ecosystem.

    - Conceptually, each option at layer n has to interoperate with every option at layer n - 1.

    - This leads to n^2 combinations and creates considerable coordination headwind.

  - If you design the system with a narrow waist–a layer that has only a single general purpose option–then you allow easy speciation above and below.

    - Layer n + 1 and layer n - 1 can have as many options as you want, as long as they can go through the singular option at layer n.

    - This diversity allows the top and bottom layers to fit whatever niche is required, giving ubiquity of coverage.

  - This pattern shows up often in successful systems.

    - The internet’s [<u>IP infrastructure has this shape</u>](https://www.oilshell.org/blog/2022/02/diagrams.html).

    - LLVM’s architecture has this shape, too: a single Intermediate Representation (IR) with numerous frontends (programming languages) and backends (target architectures).

    - My friend Dimitri has [<u>remarked</u>](https://whatdimitrilearned.substack.com/p/2024-01-01) on the power of this shape, too.

- A lot of the power of [<u>Contextual Flow Control</u>](#7hw8wtefjhp5) comes from simply not allowing network access.

  - It’s kind of wild that many sandboxes allow network access by default!

    - Moving data across a network to an arbitrary host allows the data to flow from one set of laws of physics to another, completely unknown one.

    - Code with network access might fling the data that it can see to anywhere.

  - But the value of Contextual Flow Control is much greater than just neutering network (lots of runtimes do that).

  - The value of Contextual Flow Control is the ability to reduce code to more granular chunks, whose data flows can be analyzed more precisely.

    - Less code has access to privileged resources.

  - This granularity allows a much higher ceiling on the value of combining existing modules into novel combinations, safely.

    - Allowing more code reuse in novel situations.

- Part of the challenge of code broken into modules is coherence.

  - When code is in a monolith, it’s much easier to only have coherent behaviors shown to a user.

    - Every bit of code was written in the context of other bits of code in the monolith.

    - A benefit of tight coupling!

  - But when code is broken down into cooperating modules, some of which don’t know much about the others, it’s much harder to maintain the illusion of coherence.

    - Much of the code was written not knowing what context it would be called in, and what other components it might be plugged into.

    - For example, imagine hovering over an item in a list, and the preview pane showing details of that item only updates what item it’s showing seconds later.

- It’s easy to tie yourself in knots trying to make a local first architecture work.

  - Local first is a neat solution to giving users true agency over their data.

  - But there are very gnarly problems that are solved nicely with a server architecture that need to be totally rethought in a local first world.

  - When it’s local first you can also cut some security model corners.

    - If it's local and you have only a handful of collaborators, then it's OK to run arbitrary JS from any collaborator on the data.

    - But that obviously doesn't work for a larger scale system!

    - It sets a fundamentally low ceiling on usage of the system.

    - A system that is unsafe or difficult to expand to more users has a low ceiling.

  - [<u>Private cloud enclaves</u>](https://docs.google.com/document/d/1w1RbFtk2AB1QjrmPMr3BWcrhv6uJiYzhPLQd07DN2Bc/edit#heading=h.pipyx2w5mv99) are like cutting the gordian knot.

    - The agency and control of local; the power and simplicity of the cloud.

- High quality personalization is hard to demo.

  - Imagine a system that does an extraordinary job of personalizing a map for someone.

  - You take a real example generated for one person and show it to that person as well as others.

  - To the person it was generated for, it’s magical, perfectly calibrated for them.

  - To everyone else, the result just looks bland and unremarkable.

  - The person whose context it’s based on has the “key” in their head, all of the personal experiences that help them recognize how well fit it is to them, and it “pops” for them.

  - The other people just see an ordinary result that seems bland or even random.

  - It’s not possible to demo personalization to a given person without personalizing it to them.

- A post hoc selection pressure on a true signal generates an untrue result.

  - For example, man on the street style interviews by a biased producer.

    - “But it’s real people saying that!” / “Yes, and they could be a tiny minority of the population “

  - With enough data, and enough filtering, any signal can say anything else.

  - You incorrectly think, “but it is derived from a true signal”, but the heavy-handed selection dominates the underlying signal.

  - This also happens when looking at data on internal dashboards.

    - It’s very easy to accidentally lie to yourself based on the construction of your metric.

    - “I’m just doing what the data tells me to” is a comforting fiction that implies that the synthesis and distillation of signals does not include many implicit subjective decisions.

- Sometimes a new thing is in the adjacent possible but no one else realizes it yet.

  - Imagine, in that case, a swarm of founders, who have no idea what they're doing, swarm through the adjacent possible.

  - One of them will just so happen to find the new thing in the adjacent possible.

  - It just so happens to be the single lottery winner that looks like they found it, but really they got lucky.

  - The startup didn't find it, the evolutionary process of the swarm did.

- It's easier to start new companies in a new paradigm than to retrofit a company from an old paradigm to thrive in a new paradigm.

- Building a new system is hard.

  - Instead of being able to take the foundation for granted and being able to add a single incremental extension at a time, you have to keep an approximate view of all of the moving pieces of the system in your head at the same time.

    - This is a rare skill!

  - The most important thing is finding a viable system that feels good.

    - As game developers would say: to “find the fun”

    - If you can’t find a viable thing that’s fun, then nothing else matters.

  - Instead of accreting individual hardened, well-considered layers on top of an existing foundation, you need to rough in an approximate and adaptable sketch of the whole system.

    - As long as you can squint at each piece you approximated and convince yourself there’s at least one way to harden it into a production ready thing–that there are no miracles–then it’s fine.

  - Once you have a thing that feels fun, you can start hardening.

    - Harden the parts that have changed the least in the experimentation first.

    - Then keep on hardening until you have a viable first version of the product.

  - Remember: what counts as good enough / viable might be lower than you think.

    - If you develop in the open, the motivated early users can tell you when you have something worth the pain.

- Collapsing the wave function to a precise point takes significant energy.

  - Each incremental unit of time invested to collapse a wave function gives you diminishing returns; an asymptote.

  - Once collapsed, you reduce uncertainty, but also lock in details… if it turns out you collapsed it prematurely (e.g. you now realize there are specific environmental constraints that the solution doesn’t meet), you might have locked yourself into a non-viable dead end.

  - The matrix of interlocking components set mutual constraints for each other.

  - Especially when roughing in a new system, it’s important to not collapse the wave function of any particular piece before any others, lest you back yourself into a corner.

  - At each point, collapse the wave function of the most constraining item just a bit.

    - Continue until the whole system is tightened / hardened to a viable product.

    - A breadth-first approach, not depth-first.

- When there’s not a single person driving, the system is driving.

  - What it drives for is emergent and often not what you want.

- Systems scale without limit. Individual humans have a ceiling.

  - As systems get larger, the importance of the system outweighs any hero in the system, no matter how heroic.

  - There’s only so many hours in the day for a given human being to take action.

- A Saruman style worldview has a harder time even seeing or acknowledging that systems *could* play a role.

  - They see everything through the lens of individual heroes.

  - If a given hero fails, it must mean there is an equally powerful villain, or a horde of zombies, or a literal conspiracy.

    - Zombies are NPCs, so their feelings don’t matter.

    - A dangerous mindset!

- The synthesis of the Saruman and Radagast archetypes is Gandalf: the systems hero.

  - Transform the world around you by heroically building new kinds of systems.

  - A Gandalf can create a system that transforms the potential around them.

- When you are an all powerful leader in an organization where members can be fired, it is extremely hard as the leader to get good disconfirming evidence.

  - States are unlike companies in that states, except in egregious cases, can’t fire their citizens.

    - Non-dictator states inherently get disconfirming evidence, constantly.

    - That disconfirming evidence is annoying, and can even hurt, but it makes the overall system antifragile.

  - But in companies an employee who makes too much noise is liable to get sidelined or fired.

  - Employees watch carefully and see what kinds of risks that others took led to success and which ones harm them or get them knocked out of the game.

  - As a powerful leader, even a small gesture can have massive ripple effects.

    - People will whisper about that revealed preference, and it can take on a life much larger than the leader realizes.

  - As a leader, you think you're winning more and more arguments because you're smarter, but it's actually because you're more powerful and people are more fearful of you.

  - Very powerful leaders surround themselves with sycophants, unintentionally, and believe they are ever more brave and bold, doing a better and better job even if they are doing worse.

- If you are a powerful person looking for confirming evidence, you will find it.

  - If it doesn't exist, it will be created for you by the people around you earnestly trying to do the thing they know you want.

- Founder-led companies are truly different from manager-led companies.

  - Zuckerberg has many more moves available than Sundar does.

  - Founders in founder-led companies can take the steering wheel and turn it wherever they want, and the company must follow.

    - This means that a founder can navigate an org around an obstacle it cannot see or understand.

    - But it also means that if the founder is wrong, they can crash the organization: there’s no one and no thing strong enough to counter-steer.

  - Founders have to know how to steer the organization they actually have, and if they missteer they can do a lot of damage, waste huge amounts of effort, and possibly crash the company.

  - Companies start out handling like sports cars, and slowly as they grow start to handle like big rigs.

  - You need to drive big rigs very differently.

    - A sports car you can turn on a dime; if you do that in a big rig you'll jack knife.

    - As the company gets larger, every steering adjustment cascades out through successive waves in the company, needing to be reinterpreted and ingested by teams and then transmitted onward.

    - If there's not enough slack, by the end of the chain it can be high-speed whiplash.

  - One of the best predictors of a post-PMF founder-led startup failing is when the company grows larger than any organization that any of the founders have previously worked at.

    - Being the boss is fundamentally different from being an employee, and that difference scales super-linearly with organization size.

    - You need to experience it to have the knowhow to navigate it.

    - If you haven’t been in the emergent politics of an organization of that scale, you won’t understand intuitively how they work.

- A founder who is used to only driving a sports car gets frustrated when it grows into a big rig.

  - They'll search for the villain.

  - The villain is the system!

  - There is no one to blame!

  - It’s not a conspiracy of jealous fakers.

  - It emerges fundamentally with scale.

- For large, viable systems, steer them with a light hand.

  - In large systems, a twitchy hand on the steering wheel creates huge amounts of thrash and unintended consequences.

    - Whiplash in an org from twitchy drivers is expensive (all of that movement and sprinting of huge numbers of employees) and also really hurts the employees being whipped around!

  - But in a small, not-yet-viable system, you want more of a heavy hand.

    - The most important thing to do is steer it to a thing that could become viable.

    - As a small system, it responds better to steers, with fewer indirect consequences.

  - In small systems that are not yet viable, the direct benefits of steering outweigh the small indirect costs.

    - In large systems, that reverses.

- Evolution is a flood-fill algorithm of possibility space that finds currently viable possibilities.

- On the outside, companies fight for their existence in a competitive evolutionary environment.

  - Internally, companies tend to be command-and-control dictatorships.

  - This helps ensure that don't just diffuse into bland nothingness, and have a coherent answer... even if it's wrong.

  - It is the yin and yang between coherent internally and competition externally that interactions that creates value, creates things in the world.

    - The fighting of entropy by creating coherence.

  - As competition heats up from ever-lower friction, companies turn more to command-and-control internally to counteract it.

- Why does the VC model of seeds, saplings, trees not work within an organization?

  - Organizations are about consensus.

  - Drucker: to have innovation internally you have to wall them off and cannot judge by the standards of the existing organization, and only once they could stand on their own you try to integrate back in.

  - Rule of thumb: don't aim for consensus (which leads to blandness), aim for everyone agreeing that this investment won't kill the org, and one person thinks it could be great.

  - If you try to do seed bets internally they need to have freedom to attack the host.

  - Executives often want to pull weeds as they sprout up to not cause problems or eat away at the coherence.

    - An anti-evolutionary pressure!

    - Kills the antifragility of the system.

  - Cisco has a whole spin in strategy that created external startups that would be re-integrated if they went well.

    - It was killed off partially because the winning teams undermined social trust by being "unfair"

  - Is it possible that any company that had a successful venture farm internally would tear itself apart?

  - For a company to cohere, it has to see itself as one thing.

    - For an evolutionary system to work, the things must be in competition with each other, to see each other as separate.

    - Large organizations are a tenuous balance between being “one thing” and “a swarm of competing things”.

- Strive to act in a way that you can be proud not just of *what* you accomplish, but *how* you accomplish it.

- To create high trust, look out for people.

  - Do right by them *proactively* without having to be asked.

- It’s easy to have a high trust culture in a small org.

  - It’s not hard!

  - "We're very proud of the high trust culture in our 15 person team. I don't know why larger companies can't do it, too!"

  - If you're a 15 person team and you *don't* have high trust, you're doing something very wrong.

  - The difficulty of creating a high-trust organization scales super-linearly with the number of members.

- If you aren't acting like an owner, you can't take the blame necessary to grow.

  - If you’re the owner, there’s no one else to blame for a failure.

  - You have to look inside yourself and find what needs to change.

  - When you cast blame, you abdicate the ability to grow from that situation.

- Coaching can help you improve in ways you didn't even know to ask for help with

- Disconfirming evidence need not come in a "no, but" stance.

  - It can also come in a "I wonder..." or "How might we..." form, that allows a "yes, and" engagement.

  - Not all disconfirming evidence is created equal.

  - You should share disconfirming evidence into the group, but not force the group to take it--perhaps you're calibrated wrong and it’s not useful!

  - When you share disconfirming evidence in a “no, but” modality, the group is predisposed to meet it in a defensive crouch.

    - It feels like getting a shove, or even a punch in the face.

      - The only option is to crouch defensively.

      - There’s a very real chance the group will knee-jerk reject something that could have actually improved the idea.

    - But if the group knows they have the option to absorb the evidence or ignore it, they don’t put up their defenses, and are more willing to engage with it, perhaps learning something!

- If you have a chip on your shoulder you won’t learn.

  - Learning requires absorbing disconfirming evidence, not deflecting it.

  - When you have a chip on your shoulder, you’ll be in a defensive crouch.

  - Everything has some level of BS.

  - So if you have a chip on your shoulder you’re likely to find confirming evidence for why you have a chip on your shoulder.

    - If you're in a good mood you'll see the positive parts, if you're in a resentful mood you'll see the BS primarily.

  - It's hard to reset your perspective once you have a chip on your shoulder.

  - The easiest way? Just get out of that situation and into a new one.

  - In a new situation it’s easier to be in learning mode.

- The aha moment is the emotional experience of uncertainty collapsing into certainty.

  - It's a sign of relief, of delight.

  - When you help people around you have that aha moment, you give them a toehold of certainty, something to cling to in the swirling chaos of uncertainty.

- Strong leaders can transmute anxiety into strength.

  - In uncertainty, team members can give their anxiety to the leader, and the leader uses it as disconfirming evidence to make the idea stronger.

  - The team members offload stress, the leader onboards useful information to make the project stronger.

  - The anxiety transmutes into something useful in the hand off.

- A rhetorical trick to connect with someone on a controversial topic.

  - First, stake out the two extremes on the spectrum that everyone can agree are unreasonable.

  - Then say, “we agree on that.”

  - Now the question is where in that range is the right balance point.

    - “I think X, I imagine you think Y. But these are not actually that far apart. Reasonable people can disagree!”

  - With this move, instead of the two of you being on different sides of the issue, you’re now on the same side: the side of reasonableness… and the only disagreement is a small one on matter of degree.

  - By default in charged and chaotic environments, people see the other side as a negative caricature of what it actually is.

  - This move can help negate some of that over-the-top caricaturization of one another.

- Imagine you're in the socialized mindset.

  - As a reminder, this means that your value structures are set by whatever the group you belong to thinks.

    - [<u>https://glazkov.com/adt-primer</u>](https://glazkov.com/adt-primer) is the canonical primer on these topics. It’s excellent!

  - OK, so you’re in the socialized mindset.

  - The group you've aligned yourself with gets increasingly hateful and gross.

  - But your ego is literally staked on being a part of that goup; without it your self image would shatter.

  - So as it gets worse, to resolve the cognitive dissonance you have to instead imagine the world is getting worse, and the group is simply rising to meet it and defend against it.

  - A toxic spiral, a supercritical state.

- The secret to life is things you've heard a million times already, you just weren't ready to hear them yet.