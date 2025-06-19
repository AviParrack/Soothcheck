# 3/17/25

- Claude Code is kind of like a ride-on mower with a stuck accelerator and a loose steering wheel.

  - Very powerful, and if you need a general result in an open lawn it's fine.

  - But if you're trying to do specific detail work near the flower bed, watch out!

- OpenAI's new responses API is stateful and also offers good enough default integrated tools you can use... that obviously are only available within the OpenAI ecosystem.

  - It’s a play to increase stickiness by offering something convenient that makes it harder for developers to switch to other models.

  - A monkey trap.

    - Has nice returns in the short-term for developers, but at a significant long-term cost.

  - It also makes it harder for other platforms that abstract over multiple models to offer the same kinds of features, because now users can get them for free… if they just commit to one model provider.

  - It’s OpenAI trying to change the game from stateless, easy-to-swap LLM providers, where the only competition is on quality and cost of the model, and move it into more of an ecosystem value proposition.

- Documents are alive; chats are (mostly) dead.

  - In a document, you can change any part of it at any time.

  - In a chat, you can only append new things at the end, you can’t change the parts in the middle.

  - The only adaptive part in a chat is at the far edge of it as you and the LLM append tokens.

  - If the LLM went in the wrong direction in a chat, you have to wrangle it back to a path you like, which might be hard if it has momentum in a distracting direction.

  - If an LLM interprets a doc incorrectly, edit the doc and try again.

- Apps are great for simplicity but they are a jail for your data.

  - The app model gives us simplicity but also centralization.

  - Your data can't leave, so your use cases must flow to it.

  - The app with the strongest gravity attracts everything; a gravity well.

- Tools, not apps.

  - "App" is different from an application.

  - An application is possibly general-purpose, and stores its data in the filesystem.

  - An app is a single-use bit of software, with its data model welded shut inside of itself.

    - Its own little pocket universe.

  - This makes apps safe to install but also fundamentally limited in how they can interoperate.

  - Apps are fixed, single use, little islands.

  - Even if you could generate apps on demand, it still wouldn’t fix the problems of the app model.

- Tools, not agents.

  - Agents take our agency.

  - Tools extend our agency.

- I want Just Right tools.

  - Tools that are incidental, human-scale, tied directly to my needs in that moment.

  - Not too big, not too small.

  - Not for some average market participant, but for me, in that moment.

  - Just Right.

- Building software is important for everyone, but before only engineers realized it.

  - It was so hard to do that only people who specialized in it (engineers) could even do it.

  - But nearly everyone has situated needs where having Just Right tools created on demand would help them live a more whole life.

  - If you call it “software,” lots of people who could benefit from it won’t realize they need it.

  - Another reason to call it “tools.”

- Imagine an enchanted tool creator.

  - The tool creator itself is enchanted.

  - The tools it creates are also enchanted.

- Vertical saas gives us “operating systems” for businesses. Why don’t we have it for consumers?

  - Vertical Saas is an extremely powerful business model.

  - It turns out that if you can create an “operating system” for a given niche of business use cases, it can be extremely valuable.

  - Within a given type of business (e.g. gyms) the needs are often very similar, so it’s possible to make a one-size-fits-most “operating system” for that niche.

  - But individuals in their personal life are just too different.

    - They use different tools for coordinating different spheres of their lives.

    - That multiplies for collaborating with the rest of your family and friends, who might use entirely different tools.

  - Our email, our calendar, our financial data, our health data, all of it is fragmented out in a lot of tools that don’t help us do anything productive with it.

  - It’s not possible to make a one-size-fits-most operating system for our personal lives.

  - But LLMs could allow the creation of a suite of tools perfectly bespoke to a given user, to help be the operating system for their lives.

    - A personal “operating system”.

  - To do it properly would require a bullet proof privacy model.

- Email is the schelling point where all of our cross-origin use cases meet.

  - When you visit a site or an app, you have to go to it.

  - Email is where that service comes to *you*.

  - If you go to a site and get distracted halfway through your task, you forget to complete it.

  - Email is where all of the origins can come to talk to you.

  - Email is naturally async and text based.

  - Each thread can advance on its own timeline, interleaved with other tasks.

  - Why hasn’t email been a center of our personal operating systems?

  - It’s hard to make email interactive enough.

    - On a technical basis, email is static, because it’s not safe to allow emails from a third party to be turing complete.

    - On a social basis, emails aren’t interactive, because having a real person to interact with on the other side is too expensive for most businesses.

      - What percentage of the emails in your inbox come from a “do not reply” email address?

  - How much more could you get done if every email were interactive, if you could reply to every email and get a human-like response?

- It’s wild to me that people used to program on punch cards.

  - "Wait, back in the day you programmed on PUNCH CARDS??!"

  - But people did it even though it was like crawling through broken glass because it was the only way to get those valuable results.

  - There was no better way, so you did it without thinking.

  - What are the things that are like crawling through broken glass today that we do anyway?

- The concept of “vibe coding” exploded onto the scene because we were ready for it.

  - There was a concept not well covered by existing words.

  - As soon as a word popped into existence (especially from a prominent influencer) that everyone thought was a good word for that concept, it was locked in.

  - We needed a word, so a word emerged.

- [<u>A comparison of vibe coding to folk art</u>](https://x.com/jstn/status/1900243424651206985?s=51&t=vzxMKR4cS0gSwwdp_gsNCA):

  - "I see vibe coded games as something like folk art, i.e. the value is in the creative process and not the final product . i bet there's way more people who want to make slop N64 games than play them in 2025"

  - The point of vibe coding is not sharing the result, but the process of creating.

- Vibe coding + your data is empowering.

  - Vibe coding just on its own is fun but a low ceiling.

  - Vibe coding on your most important data is a powerful unlock.

- An emerging use case adjacent to MCP: personal data workshop.

  - A place where you import your personal data to vibe code on.

  - People want it so badly that they’re crawling through broken glass to get it.

  - Using tools like Cursor plus MCP to [<u>hack it together</u>](https://x.com/KaranVaidya6/status/1897690146725839193).

  - Someone will figure out how to make it easy, and be something that can make your data come alive proactively, and safely.

- The danger scales with both the amount of data and tool use.

  - Lots of data, no tools, little danger for LLMs.

    - There might be prompt injections, but they can’t cause anything to happen.

  - No data, lots of tools, little danger for LLMs.

    - There’s lots of things that can be caused to happen, but only if you were to cause it to happen.

  - But when you have both that's when it gets potentially explosive.

  - Data is gas, tools are air. LLMs are the spark.

- Sensitive data + tool use + LLMs + app-centric security model = danger.

  - LLMs turn any text into potentially executable instructions, exploding the attack surface of traditional security models.

  - The solution isn't better sandboxing, it's reimagining how data flows. Conventional security approaches would be playing defense on the wrong field—the game has changed entirely and needs a novel approach.

  - This isn't something that can be hacked together overnight... but if someone can pull it off they'll open up a whole new universe of software.

  - Sensitive data + tool use + LLMs + data-centric security model = empowerment.

- A powerful unlock is to move from app-based security models to data-based security models.

  - In the same origin paradigm, all data flows that happen within the app are mushed together, illegible: a black box.

  - The OS can’t tell how the data flows within an app, so it has to assume it flows everywhere within it.

  - So to be safe, the OS keeps all apps from ever touching and data from flowing between them accidentally.

  - But what matters for privacy and security is ultimately the data flows.

  - If the data flows were legible to a runtime that policed data flows and made sure they were legitimate, then you could allow much more granular interaction of data, safely.

  - Today we choose between utility and privacy.

  - A data-centric model with fine-grained flow control would let us have both—making even disposable software safe.

- An architectural design to mitigate prompt injection: separate control plane and data plane.

  - Control plane never sees the precise data, just user goals and schemas.

  - It then wires together the welded together pipes for actual data (which might have prompt injection) to flow.

- A rule of thumb for when speculative execution of code is safe in the laws of physics of today’s security model.

  - You can have either network access or 3rd party code, but not both.

  - If there’s no network access (ever), then even malicious 3P code can’t do much; it can only muck around locally.

    - If there’s a robust enough sandbox, the potential damage is small.

  - If you only execute 1P code, then the code has all implicitly been verified by your team.

    - It is safe by construction because you assume that your own employees wrote the code, and they are not malicious.

  - If you want all of speculative execution, network access, and 3P code, you’d need new laws of physics.

- Alan Kay once noted that computers started off like pianos and have become more like CD players.

- Computing has become too much about efficiency, not enough about creativity.

- If you have a magical tool, you need to make sure muggles can use it.

- In the early internet you had to figure out how to use the software yourself.

  - You got practice with figuring out and tweaking software.

  - But now you just take what some billionaire gives you.

  - Similar to Toqueville’s observation of how practicing democracy in the small develops the muscle for democracy in the large.

- When did the ideal of end user programming slip from our fingers?

  - I’ve previously thought of it as the shift from PCs to the cloud.

    - Cloud gives the benefit of simplicity, reliability, and scale.

    - But it trades off control.

    - Your software running on your turf, to someone else’s software on their turf.

  - Someone I was chatting with this week localized it to the shift from the terminal to the GUI.

    - The terminal is infinitely malleable; just text you can pipe together in whatever combination you want.

    - GUIs are shrink wrapped, pixel-perfect, every pixel in a particular place.

- When you're in your house, you can move a chair, put a post it note on the wall.

  - Software is our digital home.

  - But we aren't allowed to change anything.

  - You can't put a post-it note on the wall!

  - It's where we live but it's not home.

- In reacting to a study that shows humans prefer AI generated poems, Rohit [<u>observes</u>](https://x.com/krishnanrohit/status/1899901748946555306):

  - "The real underlying problem is that humans just absolutely love slop".

- Which will win, kino or slop?

  - A key battle about who defines what good is: [<u>rockism vs poptivism</u>](https://en.wikipedia.org/wiki/Rockism_and_poptimism)

    - Poptivism: good is what people like.

    - Rockism: good is defined by the tastemakers who challenge the conventions in tasteful ways.

  - Algorithms have pulled us more towards slop, and LLMs accelerate that.

  - At some point though might the pendulum swing back?

  - This section is riffing on some ideas from Robinson Eaton.

- The "average" person in a market probably doesn't exist.

  - Similar to the old joke about the "average family" having 2.3 kids.

  - An app is designed for a market not a person.

    - A centroid implied user who might not actually exist at all.

  - This is why it’s one-size-fits-none software.

- Large fixed costs need large markets to make the investment worth it.

  - You want as many customers to collectively share that fixed cost.

  - The higher the fixed cost, the larger the market necessary to make a project viable.

- When a product has a non-trivial marginal cost but is free, then there's some ulterior motive that might arbitrarily misalign with your interest.

- If you don’t pay for the product you are the product.

  - A related observation:

  - If you don’t pay for your compute it doesn’t work for you.

- The Coasian Floor sets the lower bound of size of project a company might undertake.

  - This is Clay Shirky’s concept.

  - The Coasian theory of the firm is that finding a price has a cost.

  - You need to create a cell membrane to create boundaries between things that need to transact, and that has a cost.

    - It’s kind of like simulated annealing; everything starts off fluid, but as little bits of structure are discovered, a membrane starts developing.

    - The membrane gets stronger and more ossified to become more efficient, but at the same time less adaptable.

  - The Coasian Floor is the smallest size of project a company might undertake.

  - Software in the app paradigm has a very high Coasian floor.

    - You need to assume that a market of a given size makes sense for the fixed cost of building the software to make sense.

- To maximize combinatorial potential, I want software to return to small tools.

  - The Unix philosophy is lots of small tools that are all very good at one thing.

  - This approach works extraordinarily well, but only if there is a general purpose way of connecting multiple things into a larger assemblage.

  - In Unix, this is the pipe.

  - This allows a small number of simple applications to collectively cover a massive combinatorial space of possibility.

  - Instead of hoping to get a monolith that is somehow just the right thing you need (no features missing, no extra features distracting you), you can wire together exactly what you need.

    - The efficient size of tools in that paradigm is quite small.

  - But in the same origin paradigm, wiring across origins is, in practice, nearly impossible.

  - So you can't get combinations so you need to get chonkier monoliths.

  - Every so often you get lucky and the monolith is exactly the right size for you, but often it's far too small or far too large.

  - The process that generates the software is an emergent evolutionary search process over user demand and viable business models; it is a swarming ecosystem but the process is slow.

    - The cost of producing software also means there is a Coasian Floor below which software doesn’t get produced.

    - Software has to have a certain minimum size of market to be worth creating.

  - The result is that the vast, vast majority of combinatorial space of possible tools that a particular user might want is not covered.

  - Why not just use the Unix solution?

  - In Unix every tool is assumed to by default be local and run locally and thus have limited downside risk (especially if running inside of, say, a container).

  - When you introduce the notion of software that might run remotely, or is delivered by an origin from far away, you can't trust it to not be malicious, so a Unix pipes kind of setup is not nearly as safe.

  - To change this situation you need a new way for small tools from different origins to be able to interact safely.

  - For example, you could have lots of little UX views that all collaborate on a shared underlying reactive data model.

  - But if you could get this, you could hit a disruptive new equilibrium, suddenly bringing huge swathes of a previously non-viable part of the combinatorial space of possible software to become viable.

- I found [<u>Kill your Feeds</u>](https://usher.dev/posts/2025-03-08-kill-your-feeds/) interesting.

  - Feeds decide what we will focus our attention on.

  - They allow us to sift through way, way more information.

  - This is a necessity in the modern world where we are inundated with more information.

  - But the tradeoff of a feed is that the entity deciding where to focus our attention doesn’t work for us but rather a company who prefers we just keep watching the feed.

  - Having feeds also leads to more information being created.

  - Because we can now handle more information, more information is created.

  - Similar to the “equilibrium of misery” for traffic.

  - Add a new lane to a highway, and you get more traffic to absorb the capacity.

- Humans are easy to hack!

  - Just figure out the dopamine drip.

  - The Algorithm is great at figuring it out, emergently.

  - A dystopic outcome: locked inside a box with a super-god AI Clippy… who is dopamine hacking us.

- If you strip away work from people's day to day, what do they do with their time?

  - Meaningful things, hopefully, not lizard-brain things.

  - Software should be an extension of our agency.

  - But it's too expensive to create and so we bifurcated it.

  - It’s created by others to serve others’ purposes.

  - We’ve been alienated from our tools.

  - This misalignment has eroded us, because the software that others bother to create for us optimizes for what we want, not what we want to want.

  - If we fix that fundamental misalignment that is broken it can all be fixed.

  - The web / social media is an adaptive emergent beast that does not have your best interests at heart.

  - Be careful how much time you spend in its embrace!

- Assistance (ala Alexa, Google Assistant) used to be gated on reasoning and sensing.

  - Now with LLMs, it’s only sensing that's left.

  - That is, what things might matter to you in your current context.

  - That's both a sensor problem, but also a "user having their data in a place they feel comfortable letting the assistance layer see it."

- Why hasn't there been a trusted computer assistant yet?

  - Obviously, pre LLMs the limit was the logarithmic-returns-for-exponential-cost curve of mechanistic grammars, and there continues to be a challenge with not enough sensing.

    - To help the computer sense you either need sensors, or the user to be willing to explicitly tell the service about them--which might be a lot of work, and might be scary to do without the right security model.

  - But another thing is deeper: to trust its recommendations you have to know, deep down and authentically, that it's optimizing entirely in your interest.

  - If it was written by another entity, it’s optimizing for their interests, not yours.

- You shouldn’t just own your data, but also your software.

- I like this [<u>cool combination</u>](https://www.joelsimon.net/lluminate) of LLMs plus novelty search.

  - LLMs pull us towards the centroid; this process uses LLMs in a systematic way to skate towards interesting outcomes, not average ones.

- Most technical problems are actually socio techno problems.

  - The technology exists within a broader social context.

    - Solutions that are perfect or complete within a given system could be incomplete or broken in the broader system.

    - For example, the [<u>\$5 wrench attack on perfect crypto</u>](https://xkcd.com/53).

  - Users have to *choose* to use and adopt a technology.

  - A technology that is “correct” but doesn’t get adopted doesn’t have an impact.

  - The choice process is a social one.

  - As Vitalk has put it: [<u>The Most Important Scarce Resource is Legitimacy</u>](https://vitalik.eth.limo/general/2021/03/23/legitimacy.html).

- Diffusion processes proceed at the system’s clock speed.

  - Some of the processes are technical and go faster with better technology.

  - Some of the processes are social, and don't go faster because the clock speed is the human mind which doesn't change.

  - A common thought: "Technology moves so quickly now, so this new disruptive thing will be adopted quickly."

  - But the main constraint might be the social process of diffusion.

  - Diffusion in a system of bits is faster than diffusion in a system of atoms.

- Information takes time to diffuse through an organization.

  - In a fast changing environment, by the time it gets 7 plys deep into the organization it's out of date information.

  - The org is executing on something that is now wrong.

- Orgs focus on what's most legible, not what's most important.

  - The streetlight fallacy of organizational attention.

- If you can’t make mistakes, you can’t make decisions.

  - Decisions require making a choice in imperfect information.

  - If you have imperfect information, it’s possible–likely even–that you’ll make mistakes.

  - The fix is not to make no mistakes, it’s to make sure mistakes are quick to recover from.

  - Making decisions knowing you’ll likely need to revisit them later helps you learn.

- It's orders of magnitude faster to have the idea than to execute it.

  - Related to the friction of working with bits vs atoms.

- A superpower for making progress in ambiguity.

  - Looking at a big, audacious, amorphous long-term goal, and figuring out a thin slice that is concrete and does something useful that takes a step in that direction.

  - If you can run this loop very quickly, you can roofshot your way to the moon.

- Leadership is about clarifying goals.

  - Leadership is not just summarizing the bottom-up beliefs of all of the constituents and doing the average.

  - That average, consensus view of various plausibly viable options might not even be a viable option itself.

  - The bottom-up insights provide ground-truthing, a set of constraints to operate within and balance.

  - The same is true for setting a product vision and UXR; UXR doesn’t tell you what to do, it tells you the constraints of what’s viable.

- One of the challenges of collaborating remotely is there's a single-tracked conversation.

  - People have to queue, raise their hands.

  - In real collaborative contexts, conversations can slip into smaller sub conversations without people even realizing they're doing it.

  - They just kind of turn to someone and then if it's too loud they edge away from the main conversation just a little bit.

  - A percolating sort that creates and extends and attenuates naturally without anyone in the system individually having to think about it.

  - Totally fluid and emergent, allowing surfing the efficient frontier for a set of conversations in that moment.

  - Most tools have no affordances for it, or only weird top-down "breakout rooms" affordances.

  - Even tools like Gather require people to do something explicit to fork or rejoin a conversation, it's binary, not continuous.

- If the future is infinite software, then it has to be open.

  - A closed system can't be infinite.

- The lower layers in a system don't go away, they just get more boring.

- To get a ubiquitous ecosystem it must be open, which means that no single entity, if they went greedy, incompetent, lazy, or evil, could hold back the ecosystem.

  - That presents a hard coordination problem.

  - Crypto solves it by constructing a massive consensus machine that has momentum so high that no individual entity can divert it.

    - But that requires financializing everything to encourage everyone to participate, which is like using a caustic acid to make your coordination smooth.

  - Local-first solves it by moving everything to the edges, on user's local machines.

    - But now you have to make experiences that are resilient to eventual consistency--where "eventual" might be "not until the heat death of the universe.”

    - It’s also hard to architect collaborative features that require a single canonical state, or secret state.

    - This creates significant architectural overhead for even table-stakes features users expect.

  - Open Attested Runtimes solves the coordination problem a different way, by using Confidential Compute to remotely attest the precise structure of a remote runtime.

    - That allows everyone to verify a given host is bit-for-bit identical to what they could run themselves in a way everyone can trust.

    - If any given host ever abuses that trust, it would trivial to detect, and everyone could fail over to another host before hardly any damage is done.

    - That small exploit window makes it unlikely anyone would even try.

- Open things need to be decentralized.

  - Decentralized things are hard to coordinate.

  - Decentralized things are best when they're boring and change rarely.

  - The innovation should happen on top of a boring, general purpose, decentralized runtime.

- SSL secures data during transit.

  - Encryption secures data at rest.

  - Confidential Compute secures data during use.

- Local first load bearing party trick: software that can run without the cloud.

  - You can unplug it from the cloud and it still works which shows you own and control it.

    - “Even if the cloud endpoint went evil, greedy, incompetent, or lazy, I’d still be able to have my local island of functionality.”

  - In the same way as the Greek idea: "the person who can destroy it controls it."

- [<u>Pirate testing</u>](https://blog.oshineye.com/2010/02/pirate-testing.html) is a method of creating interoperability.

  - An independent test suite that makes it easier to have multiple implementations.

  - The implementations themselves might not even care to be interoperable, but if the community wants to use it as a schelling point, a pirate testing suite can help create and maintain alignment emergently.

- Overheard: Working on AI at Google today is like being on the web team at IBM in the 90's.

- If a movement is too much of a big tent then it becomes diluted.

  - To appeal to more people it moves towards the lowest common denominator.

  - If it's too welcoming it auto-extinguishes by diluting itself.

  - If the movement has too many purity tests then it auto-extinguishes by not reaching sufficient scale to matter.

- The term 'revealed preference' assumes that the person is perfectly rational.

  - If the system dopamine hacked me to want that thing, does that count?

- Assuming a market is winner-take-all is a self-catalysing trap.

  - If any of the competitors in the market is assuming it’s a winner-take-all market, then everyone else must, too.

  - Because if it *is* a winner-take-all market, then anyone who doesn’t play to win will be knocked out of the game.

  - It might very well turn out to not be a winner take all game, but it’s safer to play like it is, just in case.

  - Everyone playing like it’s a winner take all game (e.g. deploying significant capital to get usage) makes it so anyone who doesn’t play that way can’t compete.

  - So everyone believing it’s a winner take all game effectively makes it behave like it is one.

- Emergent processes don’t do a good job counting.

  - A few apple-related emergent process examples:

    - Which bud on the apple tree should grow into an apple.

    - Where to locate an apple orchard to have a viable business.

    - Asking a generative image model to create an image with 37 apples.

  - They don’t have a global sense of state, only local state being progressively summarized at higher layers of abstraction.

  - They can get a sense of average but not count.

  - System 1 in our minds also has this average-but-not-count ability.

- In adaptive systems, the system, in its swarming and jostling, is constantly trying lots of little variations from the baseline.

  - Most of them don't work and fade away, evaporating quickly.

    - The thing that tried them dies, or the thing that tried them sees it didn't succeed and doesn't try again.

    - Either way the thing evaporates.

  - But every so often one that is better is found; it has a consistent ground truthed bias that makes it stand strong amidst the noise.

  - This is conserved, naturally.

    - The thing that tried it keeps trying it while it works.

    - Other entities notice and also try the same thing.

    - If it's a genetic mutation, that genetic mutation gives a consistent bias to success, so it becomes more and more represented in future generations.

  - So the system is jiggling constantly trying everything, it's just the only ones that cohere and stay around to be built on are the ones that are effective, so they're all that we see as macro-scale changes.

  - This is similar to how the principle of least action arises in physical systems, as I recently learned from [<u>this mind-bending Veritasium video</u>](https://www.youtube.com/watch?v=qJZ1Ez28C-A).

    - Light waves are constantly emanating in every direction; it’s only the ones that are near the minimum that happen to interfere constructively (as opposed to destructively) and thus be visible to us.

- I love the tagline of [<u>Niche Design Zine</u>](https://nichedesign.press):

  - **"Ditch trends. Avoid playbooks. Reject boredom.**

  - Embrace the niche to rediscover authenticity."

  - Very Rockism vibes!

- A shark believes "anything I can get away with is permitted."

  - Never trust a shark.

  - When they act in a way that takes advantage of your assumed good faith, the shark would say the fault is on you.

  - "How dare you double cross me and abuse my trust!"

  - (Unapologetically) "I'm a shark, sucker! That's what I do!

- When someone's frustrated, instead of pushing back (which riles them up) or validating them (which doesn't encourage them to change), ask questions and seek to understand.

  - By asking questions sometimes you can nudge them and they unpack their own underlying problem.

  - Meanwhile you're engaging so they feel heard and held.

- Perfectionists have a harder time collaborating.

  - Because they have a mental model of how a given thing should work and their collaborators don't, so unless they have the same very high fidelity mental image of what it will be, they'll be hard for everyone else to collaborate with.

- When you're in victim mode you assign blame to your tormenter.

  - You commit a form of fundamental attribution error, with an implied "that other person is actively trying to thwart me."

  - Which is almost never true.

  - The worst thing you can do to a victim when helping them is give them the "yeah the world is out to get you".

  - The best thing you can do is coach them to see the constraints on themselves and others and help them get to a solution mindset, with an iterated series of successive "why" questions about how the other parts work.

- Change moves at the speed of trust.

  - Within an organization, change requires a period of ambiguity.

  - Moving from a thing that is known to at least be coherent to something that is new.

  - A bit of a leap of faith.

  - Leap of faiths require trust.

- External processors throw out ideas and see if they stick.

  - Sometimes those ideas just distract and randomize and make people think less of them.

    - "What a dumb idea, they must not be very smart."

  - Internal processors in contrast might self-censor ideas that would have turned out to be great.

  - Every so often an idea that is burped up is game changing.

- One of the reasons foreshadowing and calls-forward works in stories is the bisociation collapse.

  - “What is that novel thing I see in the distance? Oh, it’s that thing I already know!”

  - Gives you a little roller coaster bump in your stomach that's enjoyable and feels meaningful.

- The reason it's called the gilded age is that it was great for a very small number of very rich people and terrible for everyone else.

  - “Gilded age” is one of those terms that we learn in school and think of as an atomic unit, just a meaningless set of sounds that we associate with that time period.

  - But when you unpack the construction of the phrase you understand the deeper insight.

- Financializing things makes them more efficient but more hollow.

  - Over time the process can create gilded turds.

    - Just the shell of quality; hollow or actively gross inside.

  - When that force is turned on communities, you get a bunch of people so upset with the system they want to smash it, even if doing so would hurt themselves too.

- When my daughter is trying something new and figures it out, she gets what she calls "the sparkle feeling."

  - Adults might call that “flow state” or “being in the zone of proximal development.”

  - Optimize to feel “that sparkle feeling” as much as possible.

  - The sparkle feeling has a half life that is independent of the scope of the achievement.

    - If you achieve something 2x as big, you don’t feel the sparkle feeling for 2x as long.

  - Stretch too hard and you’ll often be frustrated and give up, and when you do succeed you’ll feel it more acutely but it will fade just as fast as it always does, so you won’t feel much more of the feeling.

  - If you don’t push hard enough the acuteness of the feeling will be less.