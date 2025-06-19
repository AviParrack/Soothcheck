# 3/31/25

A [<u>recorded conversation with Aishwarya Khanduja about this week’s reflections</u>](https://share.descript.com/view/GAWp8VRs9I1). [<u>Previous episodes</u>](https://common.tools/common-ground).

- [<u>Anthropic’s research on the inner workings of LLMs</u>](https://www.anthropic.com/research/tracing-thoughts-language-model) is fascinating.

  - They’re studying LLMs less like an engineer would study a technical artifact and more like a neuroscientist would study a mind.

  - All kinds of interesting emergent behavior about how it handles language, fuzzy mental math, why jailbreaking tricks it in some cases.

- A social sifting process is where a swarm of intentional human activity leads to macro-scale emergent phenomena.

  - No individual human's decision matters that much, the emergent process does.

  - In what conditions does a social sifting process create more quality?

    - When there's a consistent bias to the collection actions so the average of the noisy input leads to the bias popping out sharply.

    - The bias can be things like "many people seeing this would find it useful."

  - A social sifting process is already basically an AI.

    - Swarm intelligence.

  - The emergent outcome can be wonderful and better than what any individual could have produced; or a terrible hellscape that no individual wants and yet everyone gets trapped in.

- LLMs are fundamentally planetary-scale consensus machines.

  - They are large, hard-to-update social sifting processes.

    - The way to update the ranking is to retrain the entire model.

  - They learn to replicate bits of writing that humans in the past found valuable enough to write down and share.

- Every time a human makes an intentional decision they leave a residue of it.

  - Some technologies are good at summarizing and extracting that residue at planetary scale.

  - LLMs do this with writing, for example.

  - A social sifting process is one that can automatically incorporate this residue to provide better results for everyone.

- [<u>A signpost</u>](https://x.com/liminal_bardo/status/1905582947920421072): an LLM being used to update a comic about the limitations of LLMs to point out similar limitations in humans.

- Remember: treat LLMs less like oracles and more like muses.

- Insights that "sound good" are what LLMs give.

  - Which means you have to be careful, because it's giving you a consensus answer, which might be wrong.

  - Performative rigor.

- Claude Code is powerful but hard to control.

  - You can’t really see where it's going to steer it.

  - So much of what it’s “thinking” is not shown to you as a user.

  - Like an intern, it can sometimes get stuck in a corner… and then go to absurd lengths trying to claw itself out of that corner.

  - If you haven't heard from the intern in three days they're likely off the tracks.

  - A trick for Claude Code to make sure it doesn’t get stuck: “Give me two plans for how to accomplish this, and I’ll pick one.”

  - That makes sure it doesn’t just YOLO into a solution, but gives you an option to veto.

  - It also allows you to make sure it’s actually picking a good option, and considering alternatives.

- If you just YOLO writing of code, you get logarithmic value for exponential cost.

  - As you add new features an old thing pops out of place and breaks.

  - When that happens you know it’s a poorly architected system.

  - Always sprinting, never refactoring.

  - That’s what LLMs do when they write code.

  - The "always firefighting" mode of software development.

- Engineers now need to be more like engineering managers when working with LLMs.

  - How to wrangle multiple agents to a coherent result with a reasonable architecture is very different from writing code by yourself.

  - When it’s just you you don’t need to communicate plans clearly or even really have them.

  - But when you’re trying to coordinate the behavior of others, you need to communicate more clearly.

  - Previously this shift from IC to manager was a challenging phase transition, but now everyone writing code with LLMs has to make it!

- Just because there’s a ton of momentum and energy in an ecosystem of power users does not mean it will break out into the mass market.

  - There could be an invisible asymptote that makes it fundamentally hard to break out beyond power users.

- Greasemonkey had a ton of user demand, but also had a low ceiling.

  - Greasemonkey was an extension in Firefox back in the day that allowed users to install little content scripts to add features to sites they used.

  - It was powerful… but also dangerous.

    - A malicious script could steal your login credentials!

  - Greasemonkey was a power-user feature, and could never scale beyond that.

  - It required users to be a power user to install it in the first place.

  - But it also required users to be a power user to audit and get a sense of the danger of scripts they installed.

  - MCP seems similar to me.

  - Tons of momentum among power users and in the ecosystem, but fundamentally presumes a dangerous, high-friction tool.

  - A forever power user ecosystem.

- You can vibe code on your local machine, but how do you share it?

  - It has to be distributed as a web app.

  - Today most of the tools to make a web app with LLMs assume you’re comfortable on the command line.

    - That sets a low ceiling.

  - You could make it so you can make apps in the web app, without having to go to a terminal.

- The web is not just a protocol, but a medium.

  - Apps use the internet, but are not a part of the web.

  - They’re each little walled gardens, separate from one another.

  - Each stands alone, individually.

  - Contrast that with websites, which are deeply, indelibly part of the medium of the web.

  - Where one website begins and another ends is much more subtle.

- A collaborative substrate that allows executable code is open-ended and powerful... but also dangerous.

  - You can’t just YOLO it.

- MCP was clearly designed in a context without thinking through security.

  - When you use it you YOLO context to a server you just connected with.

  - But MCP has tons of momentum, with everyone jumping in on it.

  - Will there be a security catastrophe that sets a low ceiling of distribution?

  - Or will there be so much value created with MCP that the ecosystem despite (or because of) the mess, so the ecosystem figures out a way to retcon a good enough security model after the fact.

- Which happened first, ecommerce or SSL?

  - That is, before SSL, sending your credit card to a website was dangerous, but the value was significant, so some people did it.

  - There was clearly *some* ecommerce before SSL, but it exploded some time after it.

  - It’s hard to say if SSL was the limiting factor, or if it would have happened anyway.

- MCP doesn’t yet have an auth layer.

  - Anthropic is sprinting to add it, but whichever entity handles auth in a centralized way and gets more powerful will likely get a significant amount of leverage over that ecosystem.

  - It seems likely what OpenAI will attempt to do, in its embrace and extension of MCP.

- Writing software is dangerous!

  - There are tons of footguns.

    - Sharing private API keys.

    - Not protecting a user’s private information.

    - SQL injection.

    - Cross-Site Scripting.

  - If you’ve never written software and are just vibe coding, it’s easy to accidentally blow your foot off.

  - The UX of a plane’s cockpit is *intentionally* overwhelming.

    - If you are overwhelmed by all of the dials and switches, you don’t have the proper training to safely fly and should stay away.

  - But a substrate that makes it possible to vibecode on sensitive data, and also do it *safely* would be a massive unlock.

  - The right sandbox that allows lots of things will be useful.

  - But most of the sandboxes today for vibecoding are very restrictive.

  - This is partly because the same origin model is unforgiving.

- MCP and vibe coding show that there's a lot of demand for people to create tools on demand that work on their data.

  - But MCP fundamentally seems to assume

    - 1\) someone comfortable doing things on the command line,

    - 2\) in a trusted environment

      - Both trusting the client, but also trusting all of the context to not prompt inject you.

    - 3\) local

  - These three constraints set a significantly lower ceiling that prevents mass-market adoption.

  - There are ways to sand down some of the rough edges for each of those, but not remove the fundamental ceiling.

  - It’s possible that MCP has a ton of momentum among engineering types, but never breaks out.

- Vibecoding is clearly a real thing, but it's missing the substrate to publish and share the things you make.

  - Ideally it would be a shared substrate, so they aren't individual applications you have to convince people to use them.

  - I hope that the substrate that emerges is

    - 1\) open (not a closed ecosystem)

    - 2\) private (not just YOLOing data and prompt injection everywhere)

    - 3\) remixable (encouraging creativity and active participation)

- I want a geocities for vibecoding.

  - You shouldn't need a CS degree to make your personal data work for you.

  - MCP feels like the awkward adolescent phase of something much bigger.

  - I want an open, private, collaborative ecosystem for everyone to remix their digital lives.

  - “Hey come check out what I’ve been tinkering on”.

  - An arena of IKEA effects, where people feel active, creative, and connected to what they built.

  - Trying to have every person learn security is hard; so instead how can you design a substrate that makes it safe to do vibecoding, collaboratively, on real data.

- It’s possible to get good AI-generated little micro-apps, with some effort.

  - Normally a few iterations are required to find something good.

  - After some effort you’ve discovered a useful point in the latent space.

  - Ideally as the swarm of the ecosystem explores, they can automatically share save points: known good points in the latent space.

  - When you start your journey, you don’t start randomly; you draft off of the collective wisdom of the people who have come before you.

- One of the challenges of the long tail is a discovery process that finds the diamonds in the rough.

  - Ideally there's a social sifting process that is auto-catalying and gets better as more and more people use it.

- I built a chatbot into my [<u>https://thecompendium.cards</u>](https://thecompendium.cards).

  - I use the Compendium as part of my information management workflow.

  - I have over 20,000 private working notes cards in it, with hundreds added every week.

  - I also re-import the Bits and Bobs into it every week.

  - In the past I’ve found being able to pass the Bits and Bobs to Claude as background context made it a much more powerful brainstorming partner.

  - But recently the amount of Bits and Bobs–even just focusing on the ones related to my job–was far too much to fit in the context window.

  - So I added a feature to the Compendium allowing me to chat with any collection of cards.

    - This feature is only enabled for me, so other viewers won’t see it.

  - If there are too many cards to fit into context, it sorts them based on each card’s embedding’s similarity to the first user message in the chat.

  - This naturally focuses on the more relevant cards.

  - I pointed at my Bits and Bobs and asked it to write a manifesto for Convivial Computing.

    - I think it [<u>did a pretty good job</u>](https://thecompendium.cards/chat/cb3487753d093a4e)!

- A question of "would a normal person find this reasonable?" historically couldn't be done mechanistically in software.

  - So you had to do a ton of complicated and intricate things to make it so the system didn’t have to ask that question.

    - For example, the same origin policy.

  - But now it's possible with LLMs to figure out what humans would find reasonable without needing to ask a human.

- In the beginning of the web, writing experiences as a webpage was a challenge.

  - But you did it to get access to orders of magnitude cheaper distribution.

  - “Put yourself in this straight jacket, but now you can fly.”

- A lot of software today is antisocial.

  - "Hey, I was thinking... what if you watched just one more video about this terrifying viral thing that will make you more anxious?"

  - If a friend did that to us they'd no longer be our friend!

- When the costs of production drop, the Coasian Floor drops.

  - The Coasian Floor is the lower bound of where it makes sense for any company to execute on a business opportunity.

  - That floor is not some fixed bar; it is an equilibrium based on the cost of inputs, the value to be created, etc.

  - A disruptive drop in the cost of key inputs has the potential to be a massive change in the Coasian Floor.

- There's an entire universe of cozy personal use cases that are not individually venture-backable.

  - Each use case is too small of a market, or too small to monetize.

  - Those use cases are below the Coasian Floor.

  - But the entire universe as a whole, if it could be catalyzed, would be venture backable.

- What if you had a competent and indefatigable intern to build you software for your family life?

  - You could solve all kinds of cozy problems that were below the Coasian Floor.

  - You’d have to trust that intern to not leak your sensitive information, actively or accidentally.

- Imagine: Deep Research, but the output is a perfectly bespoke app that solves your precise need in this moment.

- If people make tools they themselves think are cool, and the only people who can make tools are engineers, you get a lot of tools for engineers.

- I don’t want a todo list, I want a *do* list.

  - Items that are expressed clearly enough can simply be executed by the system.

  - Execution is hard because of all of the little unexpected details that show up that can’t be handled mechanistically and require a bit of judgment.

  - But LLMs can take over once you get to the point where they’re specified well enough that the LLM can handle the ambiguity.

- “Good enough” is relative to the alternative.

  - If there is an established alternative that is effective, the bar can be quite high.

  - If there is no effective alternative, then the bar can be quite low.

- Don’t automate important tasks users do today; automate things they don’t bother with today.

  - If you automate important tasks they do themselves today, anything less than perfect could be below the bar of “good enough.”

    - If it fails even once, the user might learn they can’t trust it.

  - But there are likely some use cases that the user would get some value out of but it’s not worth the effort for them today.

  - If you can automate those, then you create new value that wasn’t possible before.

    - If you fail, the user is no worse off than they are today.

    - All upside, no downside.

  - This is similar to Ethan Mollick’s frame on LLM quality thresholds of “best available human”.

    - Don’t compare an LLM’s quality to the expert in the field, compare its quality to what the actual human who would have done it instead is.

    - This is often a much lower bar.

- Pipelines are very sensitive to the weakness of their components.

  - The chance the entire pipeline works requires multiplying the likelihood of any of the subcomponents failing.

    - All it takes is one component in a pipeline not working to invalidate the whole pipeline.

  - With a high failure rate of components or a long pipeline, that error rate of the pipeline goes up very quickly.

  - Components that are larger and take on more tasks are more likely to have a high failure rate: to have inputs they can’t produce good outputs for.

  - The more targeted the inner components; the more hardened and successful in the past they’ve been, the more likely the entire pipeline works.

- Doing general purpose extractors is hard, a logarithmic curve of value for exponential cost.

  - Anyone who’s written a web scraper knows this in their bones.

  - No individual case is hard, and easy for someone to say "the fix for this case is easy."

  - The problem is that all of the individual cases are below the Coasian Floor of being worth tackling individually.

    - Also often all of them are quite different, so solving one doesn't necessarily help with the others.

  - To tackle general purpose extractors in a sustainable way requires a social sifting process, where the savviest users, in solving their own problems, produce solutions that everyone else can draft off of, too.

- If you can’t build a piece of IKEA furniture, the problem is likely you, not the furniture.

  - When you're constructing a model kit, and something isn't working, how do you react?

    - Is the quality of the components poor and they don't fit like they should?

    - Is there a missing piece?

    - Are the instructions incorrect or missing steps?

    - Is it something you did wrong?

  - If you're building an IKEA item, you know the problem is you, not the quality or components or instructions.

  - That's comforting!

  - When the problem might be the components, you give up in frustration more often; there’s nothing you can do to improve them.

  - But the problem is you, you can focus and try again.

- Palm Pilot’s Graffiti input system likely made users more willing to put up with lower quality.

  - Back in the day, Graffiti was a special input system that required you to trace specific shapes with your stylus to input specific letters.

    - Many shapes were the same as the letters they encoded.

    - But some were wildly different.

  - Instead of saying, “just write like normal and we’ll figure it out”--an impossibly high bar to hit even now, let alone in the 90’s–they said “learn this new way to input that is inspired by handwriting”

  - If the system didn’t understand your input, it wasn’t just “the system is dumb,” it also had a “oh I didn’t make the shape properly.”

  - The outcome felt co-created by the user and the system, which made users more willing to put up with failures and not give up in frustration.

  - There was something the user could do to meet the system where it was and get higher quality output.

- I want a tool that detects when the stars have aligned.

  - There are some times that the stars align.

    - For example your friend who you haven’t seen in awhile happens to be free and in town on the same weekend you are.

    - Or your favorite band from your college years is playing at the local venue.

  - It’s extremely hard to detect these things have happened.

  - You need to keep track of dozens of calendars or subscribe to dozens of mailing lists you rarely read.

  - Having too many calendars on screen at once is overwhelming, so you likely don’t check the “FYI” calendars proactively very often.

  - It’s also hard to write mechanistic rules that would tell you that something interesting that you might want to pay attention to has happened.

    - To do so requires a level of human-like judgment.

  - But now with LLMs you can set a high-level, human intent, and have an intern who never gets bored who can keep an eye on things for you.

  - The rules for the kinds of things you’d want to be notified of could be quite simple to express, if you could assume human-level understanding and base judgment.

- The quality of a cherry picked thing doesn't reveal anything about the underlying distribution of quality.

  - It only reveals something about the quality of the curation function: the quality of the cherry picking process.

  - If you don't know how much effort went into the cherrypick, it could be like the magic trick where the magician just puts orders of magnitude more effort into it than you thought would be reasonable.

    - "Wow, the demos are so good, this system must be great!"

    - "But how much effort did it take to find a thing to demo that was that good?"

  - People judge the quality of the magic trick based on the impressiveness of the result, but it's really more about "impressiveness per unit effort" that matters.

- The willingness of a user to evangelize is tied to wow moments more than the total value.

  - It’s possible for a tool to deliver lots of value but with lots of small savings constantly.

  - That’s a product that a user might love and be unwilling to give up, but probably wouldn’t actively evangelize to their friends.

    - The small savings are nice but aren’t worth writing home about.

  - Contrast that with a service that has the same amount of value, but more spiky.

  - Rare spikes of standout “wow” moments.

  - Those wow moments are prominent, easy to describe to others, and hard to ignore.

  - That makes them more obvious to evangelize to others.

- The goal of a platform is to reduce how much effort developers have to invest to accomplish things they care about.

  - It’s mainly about reducing effort for things that developers are already doing.

    - If they’re doing it even though it’s hard, they definitely want to do it.

    - If they’re not doing it, then it might be because it’s too hard… or because they don’t care about it much.

  - Real platforms do this by watching what developers crawl through broken glass to do and then making it so the next person with that use case doesn’t have to crawl through so much broken glass.

- Resonant things are things that the closer you look, the more compelling they become.

  - Most things the closer you look the less impressive they become.

- Once you have a dishwasher you can’t imagine living without it.

  - Modern software generates a lot of dirty dishes for us to handle!

- Semantic Diffusion: popular terms tend to dilute and lose their meaning.

  - This was described by Martin Fowler.

  - Every time that someone uses a term, there’s some non-zero chance they use it “wrong” or imprecisely.

  - That means the more people who are using it, the more likely that the term regresses to the mean.

    - Sliding from its original distinctive meaning to something more generic.

  - The more popular the term is, the faster the game of telephone goes and the faster it happens.

- Every action is more likely than not to regress you to the mean.

  - So the more activity and movement in your system, the faster it regresses to the mean unless there’s a countervailing force that is stronger.

- Choosing to use a word is like kicking a hackysack into the air.

  - Each use of a word is a new vote that the term is worth keeping around and using.

  - Terms that no one bothers to use fade away, captured (maybe) in fossilized text.

  - Words that were fossilized in text in the past might one day be read by someone new who finds it useful and says it again, but it’s unlikely.

  - Words that were never fossilized and are no longer used simply fade out of existence.

- Modifying a working thing is orders of magnitude easier than creating something new.

  - You know it roughly works, and you’re just tweaking it.

  - Versus starting something new which is planting a flag in previously unknown territory; there’s a good chance the thing isn’t viable, and you have to grope around in the dark until you find something that is.

- The viability of an idea can’t be known until it’s executed and works in real life.

  - Execution is like touch.

  - You can only touch one thing at a time, right in front of you.

  - What seems like it could be viable in theory is like sight.

  - You can see many things at once and from far away.

  - It feels like looking for targets is like sight but it’s actually more like touch.

  - Exploring a fitness landscape in the dark.

- Headless things are hard to debug.

  - Nothing to see to help discover what’s wrong where you can notice what’s off with a quick glance.

  - You instead have to poke at them in the dark.

  - Imagine how hard hide and go seek would be if pitch blackness!

- A general purpose tool can't have niche features crudding it up.

  - That's why you get the tyranny of the marginal user.

  - As you grow your audience more and more features become distracting to more and more users.

  - Now every feature you add has to be massive from the beginning, it's not allowed to start small.

  - So you spend tons of time flailing to find a big thing, or launch some massive, poorly conceived thing and force users to use it.

  - Meanwhile you keep eroding interesting features out of the product to make everything smoother for larger audiences.

- The difference between backup and export: a backup is available even if the service goes away unexpectedly.

  - If the company goes down before you export, you're screwed.

  - If you can do backups and also run the software yourself because it's open source, then your use case can survive the death of the company that created the software.

- Credible exit doesn’t require users to actually leave.

  - It just requires that they credibly *could*.

  - A credible exit is effectively an ecosystem BATNA (Best Alternative To Negotiated Agreement).

  - It keeps pressure on the platform to be well-behaved and continue innovating, because if they don’t, people could leave.

  - Just because it’s technically possible for a customer to leave, doesn’t mean it’s *practically* possible.

  - That’s why it’s important that there’s a *credible* exit.

- Crypto has auto-catalyzing bug bounties with no limits.

  - If there’s a bug in your system and it’s deployed, then people in the ecosystem will take advantage of it, with no limit.

  - In some ways this is confidence-inspiring; if it’s not happening even for heavily used components, that’s a good sign the component is reasonably resilient.

  - But it also makes it extremely hard to ship any new features; each new feature has to be rigorously tested for bugs before it is rolled out in the wild.

  - Another reason crypto is such a challenging environment to operate in!

- Communicating the right “why” is worth 1000x of communicating the right “what”.

  - The why sets the context to fill in the gaps coherently automatically without having to describe it in detail.

  - Without the right context, you need a lot of precise detail to pin everything down.

  - You can fight default-diverging processes with a ton of detail, but you'll never get anything more than sub-linear returns.

- When you're in the details, you get sucked into them more; because solving details is like solving a puzzle.

  - You get a little dopamine hit each time a piece snaps into place, even if the piece isn’t important in the grand scheme of things.

  - But if you lose track of the "why" then you can easily get lost, going down rabbit holes, chasing increasingly convoluted things that don't matter.

- The more efficient the system, the harder it is to experiment with something that doesn't align.

- Innovation moves up pace layers over time.

  - As a given pace layer gets to the late-stage, where the momentum has coalesced around the obvious winners, competition moves up the stack to new layers on top.

    - The pace layer slows down as competition decreases and it moves to steady state; a new pace layer with faster rate of innovation emerges on top.

  - Before a layer coalesces and loses its energy, it absorbs all of the energy and new things can’t be built on top of it yet.

  - Things can only be built on top of stable foundations.

  - Fast pace layers are not stable foundations.

- Vibecoding will lock in today’s popular libraries.

  - There’s already a preferential attachment effect for libraries that are popular today.

    - All else equal, it makes sense to use the thing that others are already using.

    - There’s more likely to be documentation, other compatible libraries, bugs are more likely to have been discovered and fixed, etc.

  - LLMs are already more likely to recommend popular libraries.

    - They’re more common in their training set.

  - With vibecoding, you just accept whatever code the LLM gives you.

  - That allows creating lots of new stuff quickly… but also means that it will be harder than ever before for new libraries to break onto the scene.

  - Maybe that’s fine; the javascript industrial complex arguably already has orders of magnitude too much innovation/churn.

  - Perhaps front end development will now just be frozen as whatever was popular at the end of 2024.

  - As that layer gets less innovation, maybe innovation will turn to higher layers.

- Be together, not the same.

  - This was an old Android marketing campaign from the mid 2010’s.

  - But I like the sentiment!

- In simulated annealing the temperature goes in one direction: it goes down.

  - At any given point, if you increase the temperature, you’ll lose a known good location to a possibly worse one.

  - That causes you to cling to the known quantity over the unknown one.

  - When you get to the top of the curve, you stay there until you die.

  - The only way to find a new maxima is to start a *new* thing.

- You can have anything, you just can't have everything.

  - If you focus and cut extraneous things you can make just about anything happen.

  - But to do so requires cutting, which locks you onto a particular path.

  - Hopefully that path turns out to be viable!

- I found [<u>this take on trust and Promise Theory</u>](https://mark-burgess-oslo-mb.medium.com/trust-and-trustability-6eeb89df1974) thought provoking.

  - "Trust is a policy for deliberate inattentiveness. Mistrust is proportional to our sampling rate for checking up on an agent. Trust is therefore a strategy for saving effort, given finite resources."

- A tension: to be or to do.

  - This tension was described by John Boyd.

  - "To Be Somebody" - focus on climbing the hierarchical ladder, gaining status and recognition.

  - "To Do Something" - focus on making meaningful contributions and improvements, even if it means challenging the system.

- No matter how big the challenges you’ve faced in the past, in all of them you’ve survived.

  - This is tautologically true; if you’re thinking the thought, then in the past you have by construction so far survived.

  - It doesn’t mean you’re invincible, but it does mean that you have a pretty good streak of making it through even the worst that life has to throw at you.

  - Similar thought to the idea in The Street’s [<u>On the Edge of a Cliff</u>](https://www.youtube.com/watch?v=yc9gIzRhrvY).