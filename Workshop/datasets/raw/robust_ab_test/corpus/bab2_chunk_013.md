# 3/24/25

A [<u>recorded conversation with Aishwarya Khanduja about this week’s reflections</u>](https://share.descript.com/view/0r0rS0Bjcvg). [<u>Previous episodes</u>](https://common.tools/common-ground).

- LLM coding assistants are best thought of as interns.

  - You’d never have the intern design the architecture and API of a project.

  - But you would have them do the implementation.

  - That's the best way to use LLMs for code, too.

- Programming projects that used to require two hours of planning and two days of execution now take three hours total.

  - You can stay at a much higher level of abstraction.

  - But if you stay at that level of abstraction, without reviewing the code carefully, the project will get increasingly gnarly and harder to modify and extend.

  - LLMs do a better job fixing bugs and extending code than improving or refactoring it.

- LLM-written software will mostly be net-new software that otherwise wouldn't have been written.

  - It will help make it more efficient to write software.

  - But it will also allow a new class of software that previously was below the Coasian Floor and not viable.

- I agree with Simon Willison’s [<u>take on vibe coding</u>](https://simonwillison.net/2025/Mar/19/vibe-coding/).

  - "I don’t want “vibe coding” to become a negative term that’s synonymous with irresponsible AI-assisted programming either. This weird new shape of programming has so much to offer the world!

  - I believe everyone deserves the ability to automate tedious tasks in their lives with computers. You shouldn’t need a computer science degree or programming bootcamp in order to get computers to do extremely specific tasks for you.

  - If vibe coding grants millions of new people the ability to build their own custom tools, I could not be happier about it."

- Some vibe coding will be about the process itself.

  - The point is the building.

    - Programming as an intrinsically enjoyable activity, like solving a sudoku puzzle.

  - But some people who vibe code will do so in order to achieve a goal.

  - It won’t even feel like coding.

  - Especially if the software is never really foregrounded, they might not think about the software much.

  - The creation of software will be incidental, achieving the goal will be what’s important.

  - They won't even realize they’re conjuring up software.

- Systems should have a mix of squishy and hard things.

  - Squishy things allow adaptability.

  - Hard things allow dependability.

  - If you have squishy, ad hoc assemblages of squishy components, nothing works and it’s hard to tell what’s wrong.

  - The hardness of a thing is how many times it’s been used in the past successfully.

  - If you could summarize the collective wisdom of the ecosystem’s users and past experiences, you could rank components by their hardness.

  - That would allow the rest of the system to be properly squishy, safely.

- Software today is about UI before data.

  - The data is implied, not seen directly.

  - But the data is most important.

  - In a world where software is cheap, the software should fade away, be incidental.

  - The UI should be something that you can ignore or recombine.

  - The data should be primary.

- I want personal software.

  - Software just for me, just in that moment.

- Prosocial software is software that is oriented towards collaborating with and supporting humans to achieve things they find meaningful.

  - Prosocial software would function more like a supportive tool or partner, helping humans accomplish what matters to them while respecting their agency, values, and wellbeing.

  - It's software that works for humans, not the other way around.

- Composable software gives you permission to play.

  - To recombine, to experiment.

  - Developers can compose software themselves by pulling from github.

  - But today normal users can't.

    - A "one app at a time" mentality.

  - The software should be a thing that users believe they can change.

  - That belief is the most important thing.

    - If they don’t believe, they won’t even try.

  - Users should believe they can pop the hood on their software.

- Models that very rarely hallucinate will bite harder when they do, because you won't be expecting it.

- I liked this article that posits that LLMs are best seen as [<u>cultural and social technologies</u>](https://henryfarrell.net/large-ai-models-are-cultural-and-social-technologies/).

  - LLMs are more like a market or a bureaucracy than a person.

  - A social technology, not an agent.

  - That is, an emergent phenomenon that is more than just the sum of its parts.

- I liked [<u>LLM-generated code is like particleboard</u>](https://so.dang.cool/blog/2023-12-30-llm-generated-code-is-like-particleboard.html).

- Amelia Wattenberger’s [<u>new essay</u>](https://wattenberger.com/thoughts/our-interfaces-have-lost-their-senses) is absolutely gorgeous!

- I thought the analysis in [<u>Code is the new No-Code</u>](https://lumberjack.so/p/code-is-the-new-no-code) was intriguing.

  - In the past, tools that wanted to give turing completeness to non-engineers went out of their way to hide code from users.

    - Inventing UX like nodes and wires, or other domain specific languages.

    - These tend to be confusing to everyone–programmers don’t understand them because they’re DSLs they aren’t familiar with, and non-programmers get just as overwhelmed.

  - But most code is quite simple, if you could cut out all the other cruft necessary for error handling, imports, etc.

  - LLMs allow you to focus just on the key parts of the code, not all of the distracting parts.

- Demoing a tool and using it are fundamentally different stances.

  - Demoing is about meeting the tool where it is.

    - Using the tool as an end.

    - If the demo works it’s upside.

    - "What would show off this tool and make it seem valuable, while avoiding bugs or missing features that would demonstrate its limitations?"

  - But real use is about using it as a tool to accomplish things you care about.

    - The tool is entirely a means.

    - If the tool doesn’t do what you want it’s a bad means.

    - You discard means that don’t work.

  - A threshold is cleared when a team building a tool starts using it for real use, not just to demo it for themselves.

  - It’s similar to the difference between book knowledge and experiential knowledge.

- My definition of ‘tool’ in the LLM context is 'turing complete code running elsewhere that an LLM can invoke'.

  - 'Elsewhere' as in 'outside whatever sandbox the chat is executing in'.

- Are spreadsheets a good comparison for LLM adoption?

  - Simple to start, but actually not easy to learn and a high ceiling if you’re willing to put in a lot of effort.

- Is AI the product or the input to the product?

  - I think AI as transistors is the right comp.

    - Sam Altman [<u>apparently agrees</u>](https://stratechery.com/2025/an-interview-with-openai-ceo-sam-altman-about-building-a-consumer-tech-company/#:~:text=My%20favorite%20historical%20analog%20is%20the%20transistor%20for%20what%20AGI%20is%20going%20to%20be%20like).

  - AI is not the product for any but the most advanced users.

  - AI is the input that allows new kinds of products to be made.

- A thing that is the trusted assistant for your life can't also be supported by advertising.

  - It's a massive conflict of interest.

  - The problem is primarily when the assistant gives you a single answer.

  - When the assistant gives you multiple answers to choose from, and flags that some were sponsored, that is less bad.

- The compounding domino model of human interaction.

  - A principle: every action that could have irreversible side effects outside the system must be initiated by a human.

  - This principle prevents automation run amok.

  - At the beginning, this is extremely limiting; the human must constantly be in the loop, even for mundane tasks.

  - But you can add layers of leverage that then actuate the layers below.

  - As the user gets a better handle on the quality and correctness of a given UI, they can decide to add a layer that actuates it, too.

    - So for example you hit "Send all" and it hits send on 10 emails in that workflow.

    - This gives additional leverage.

    - This can continue for many layers.

  - The key thing is 1) the human always kicks off the chain reaction, and 2) each new layer is one the human chose to put in place, with calibrated confidence in the interactions of the layer below.

  - This model is simple but can lead to compounding amounts of leverage.

  - A real-world analog is the compounding domino demonstration at science museums.

    - The first domino is normal sized.

    - But each successive domino is 10% bigger.

    - By the 10th domino, it’s the size of a door.

    - The user knocks over the small domino, and it sets off a chain reaction that knocks over the door-sized one.

  - I originally heard this frame from Scott Jenson.

- I want a tool to work with my most sacred data.

  - My sacred data is personal, powerful, precious.

- Jailbreak your data.

  - I love how subversive it sounds.

  - Sounds like civil disobedience.

  - It's *your* data, so it should work for you.

- Imagine a gallery of demos that run on your data.

  - Typically that kind of flow has a “store” where you browse static descriptions of things and then choose to install them and see how they work on your data.

  - But with a data-focused security model, you could safely execute them on your data, no install step necessary.

  - Much lower friction to try anything new.

  - See things you like?

  - Keep them.

  - Things you don’t care for?

  - Just ignore them and they fade away.

  - But some of the stuff you see you’ll probably like even if it wouldn't have been obvious without seeing it on your data.

- Stateless things aren't sticky.

  - Stickiness typically shows up with some kind of state being maintained in the system.

  - There’s a weaker form of stickiness that comes entirely from a user’s familiarity with the tool, but it’s minor.

  - Generally stickiness scales with the amount of useful state stored.

- All of the strategies I find interesting harness open-endedness.

  - Open-ended systems grow at a rate faster than your own individual investment in it.

    - They are auto-catalysing, and don’t have a ceiling.

  - Open-ended possibility means open-ended upside.

  - It’s not enough to have an open-ended system, you also have to have a complement to it, so the more energy the open ecosystem has, the better you do.

- Systems need noise to be able to adapt.

  - Imagine a bullseye that lots of archers are trying to hit.

  - Each time an arrow connects, it generates a little burst of light.

  - It’s natural for a team to try to optimize the accuracy of their archers.

    - Share best practices from the best archers to help improve others.

    - Cull low-performing archers.

  - This pull towards more efficiency is the most obvious thing in the world.

  - But now imagine the lights turn out, everything is totally dark.

  - For a while, the archers continue hitting the bullseye, and when they do, they see the light.

  - But then, all of a sudden, the light disappears–the arrows aren’t connecting.

  - Unbeknownst to you, the target has moved.

  - How do you find it again?

  - You have to probe in the dark, sending arrows randomly to try to find a hit.

  - If the target continues moving, you might never find it.

  - If you would have had some noise in the arrows, some spread around the bullseye the chance is that one of the arrows would have kept hitting.

  - That would have shone the way for the other archers to update their aim.

  - This noise fundamentally allows sensing in the dark.

  - The “roving bullseye in the dark” is what actual targets are like in real environments.

  - A formal analysis I’ve seen has shown that the optimal amount of noise is proportional to the expected rate of movement of the target.

    - This makes sense intuitively; with enough noise, you have some likelihood of one of the arrows still hitting even though the bullseye has moved.

  - It’s easy to forget in real life that the target is actually roving in the darkness, but you must never forget.

  - The “bullseye” that we can see is not the real target, it is a proxy for it.

  - It makes us forget that the bullseye we see can be misleading.

- When optimizing, a metric is like a lighthouse in heavy fog.

  - It’s a proxy for the underlying reality.

  - When the lighthouse is in the distance, it’s a good thing to sight off.

    - It makes sure you’re pointed in the right direction and not getting blown off course.

  - When the lighthouse is close, if you track it too closely you will crash in the shoals.

    - The lighthouse is on dry land, as you get asymptotically closer you will run aground.

  - In metrics, as you get asymptotically closer to the goal, you get dangerously lost, chasing things that don’t matter.

  - The metric is always a proxy.

    - When you’re far away, the proxy is closer to the real endpoint than other things, helping point you in the right direction.

    - When you’re close, the proxy is farther away from the real endpoint than other things, pulling you away from the right direction.

  - Metrics need to be in the middle-distance to be true guides.

- An alternate frame of Goodhart’s law: you can either steer the ship or understand the ship.

  - The uncertainty principle applied to organizations.

  - I heard this formulation from Ade Oshineye.

- Systems that are over-optimized get hollowed out.

  - Superficially they are thriving, doing even better than ever before.

  - But inside they are hollow; zombies shuffling forward.

  - Power is centralized, everything is overly efficient, adaptability is lost.

  - A system that is alive can adapt, plan multiple steps ahead.

  - A system that is a zombie can only shuffle forward and plan a single step ahead.

- I liked this [<u>frame about public digital infrastructure</u>](https://www.cjr.org/special_report/building-honest-internet-public-interest.php):

  - "If the contemporary internet is a city, Wikipedia is the lone public park; all the rest of our public places are shopping malls—open to the general public, but subject to the rules and logic of commerce."

- Cultural shelling points (e.g. Spongebob Squarepants) emerge from good enough content repeated mercilessly.

  - The general wisdom in marketing is someone has to see it seven times before they’re willing to engage with it.

  - When you see it enough, your brain thinks “this is common enough to be worth making a handle for.”

- Filter bubbles are auto-intensifying.

  - Even if it starts out with a small random bias, that compounds as you focus more and more on signals that fit within what you already think, which is tied to what you’ve already been exposed to.

  - How much do the companies control the algorithm and how much does the algorithm just fundamentally emerge out of the structure?

  - Engagement is the emergent thing to optimize for in media; in infinite content and finite viewing time, the zero sum thing is about more watch time.

    - If you don't optimize for it, then you are outcompeted by the entity who does.

  - Our ranking signals all overfit, at a compounding rate.

- The filter bubble shows up even with nothing nefarious from anyone.

  - There's simply too much information; you must choose what to look at.

    - You must filter.

  - Filtering can't be done only by you, because it requires looking at all of the information in the first place, it must be something outside you.

  - We think we know what is true and what deserves attention, so we take our limited attention and train it on that bullseye, becoming more efficient at it.

  - But it's a roving bullseye!

- Chaos tends to multiply.

  - Anything combined with chaos is chaos.

  - How do you make it so *order* propagates?

  - You need a percolating sort; a thing that uses some energy to ground truth and rank so the good things naturally survive to the end.

- Why are government forms so confusing?

  - One reason is because as long as it's possible to file them, the agency that requires them doesn't care about improving them.

  - There's no incentive to lessen friction for users, because users have to fill them out if they want the service.

    - There’s only the single “provider”.

  - Contrast with something in industry; no one has to use this offering vs a competitor, so there's always an edge over competitors to get from making your own offering lower friction to use.

- You don’t need a theory of mind for something that your OODA loop is orders of magnitude faster than.

  - Startups' OODA loop is orders of magnitude faster than big companies’, purely based on how many people have to coordinate and downside magnitude.

- An aggregator is like a platform but it isn't one.

  - A platform has an open-ended ecosystem on top.

  - An aggregator is a platform but with a closed ecosystem on top that is controlled by the aggregator.

  - The aggregators’ platform doesn't do anything outside the context of the aggregated user experience.

- Fiction is easier to differentiate than the news.

  - The exact form of a piece of fiction is the end in and of itself.

  - News is a means: fact conveyance.

  - Fact discovery, especially in time sensitive contexts, is expensive and also hard to defend other outlets just reposting the same facts you discovered.

  - Stratechery has long talked about the Smiling Curve in media: as distribution friction declines, the hyper-niches and the hyper-schelling points thrive, and everything in between withers.

- I was talking with someone from another industry who was gobsmacked at how prevalent open source is in software.

  - It has been so long since I’ve viewed open source software as anything but completely inevitable that I was surprised by their surprise.

  - I asked ChatGPT to prepare a Deep Research report on why it is and thought it did a [<u>pretty good job</u>](https://chatgpt.com/share/67ddf107-4ba8-800e-abc2-f11bb0b2a9c7).

  - The non-rivalrous nature of software and power of abstraction seem very relevant.

- In software, consumer contexts often have winner-take-all but B2B rarely does.

  - Consumer contexts are much lower friction, so aggregation and network effects play out faster.

  - B2B contexts are paid, which require contracts, onboarding, etc: friction.

  - Consumer contexts flow quickly due to typically being free, which means they have a monetization problem.

- Aish [<u>shared</u>](https://docs.google.com/document/d/1GrEFrdF_IzRVXbGH1lG0aQMlvsB71XihPPqQN-ONTuo/edit?disco=AAABgAsZG04) how she and her friends used Google Docs in high school.

  - They weren’t allowed to use chat apps during the day, but they were allowed to use Google Docs.

  - The teacher thought everyone was writing, but they were actually using a shared doc as an ad hoc chat.

  - It was a doc everyone could edit; different “chats” could happen at various places in the document.

  - A shared canvas where chat was just one convention you could use.

  - I love the emergent social exaptation of a general purpose shared substrate.

  - Not too dissimilar to how Wikipedia is coordinated with every page a wiki with social convention constraining how it should be used.

- I liked [<u>stamina is a quiet advantage</u>](https://kupajo.com/stamina-is-a-quiet-advantage/).

  - “While stamina is the ability to sustain focused effort despite pain or discomfort, you should also think of it as the ability to stay true to your values and commitments — to hold fidelity to a worthy purpose — especially when it’s hard to do so.”

- Premature decentralization diffuses motive force.

  - Decentralization on its own doesn't produce momentum... in fact, it can *prevent* momentum from ever showing up since it's a diffusing kind of energy.

  - If you want to change something to pivot/adapt to how real people are using it, you have to convince a committee, instead of being able to just make the change and seeing how users react.

    - It significantly slows your OODA loop.

  - Decentralization enables ubiquity (everyone is willing to participate since no one actor could control or extract), but it also saps adaptability when finding PMF.

  - The best pattern: do something in the open but not decentralized (tell other people but don’t explicitly try to get them to coordinate), find PMF, and then welcome other people who choose to coordinate once it already has momentum.

- I want cozy finance tools.

  - There’s a whole category of [<u>banking apps for kids and teens</u>](https://youngandtheinvested.com/banking-apps-for-kids-and-teens/).

  - Mint was great, but shuttered after receiving no investment for a decade.

    - There was no business model.

  - The companies that might plausibly do cozy finance tools have an ulterior motive: some high-fee financial product.

- Group conversations are more coherent with a referee.

  - The referee doesn't have to be the primary voice, just the person everyone agrees is allowed to make calls.

  - When there isn’t a referee in a group, the power dynamics are unsettled; as everyone feels the lack of momentum, everyone bustles for the lead role, and everything gets frenetic.

  - The loudest voice wins, not the one with the most legitimacy from the group.

  - The referee is the gardener of the discussion.

  - They help make sure it stays aligned with the long-term interest and goals of the group.

- All creation is production plus taste to curate to a subset.

  - The faster you can run the loop the more value you get out.

  - The determinants are the speed of production of ideas and the quality of the taste.

- Our minds only perceive contrast.

  - When everything’s the same, it fades together and becomes invisible.

  - The most mundane things just don’t look like anything at all.

  - Mundane things are what make nearly everything hard to effect in practice.

  - If you underestimate the importance of mundane things, you’ll constantly underestimate the amount of effort to get to completion.

  - You’ll constantly move onto the new thing before the old thing is finalized.

- A compelling thing is something that is totally surprising and also makes perfect sense.

- When things invert, there’s an infinitely weird moment in the middle.

  - Like when looking at yourself in a spoon.

  - From far away, you’re upside down.

  - From close up, you’re right side up.

  - The point where your face flips from upside down to right side up is infinitely stretched.

- All new things start small.

  - Some start small and then can grow to eclipse everything else.

  - Some start small and stay small, adding value only in their own little pocket of enthusiasts.

  - When your valuation or scale is too big you can’t have a small thing to start, you have to act like it’s a big thing to start for investors or leadership team.

    - A dangerous kayfabe.

- It’s easier to grow strengths than areas for development.

  - You already have something that mostly works.

  - You have momentum to work with.

- If you "yes, and" what others do, you’ll overestimate how much they understand and are aligned with you.

  - "They agreed with me!"

  - "No, they just didn't *dis*agree."

- We judge ourselves by our intention, and others by their actions.

  - This is the fundamental attribution error.

- A model for the emergence of trust:

  - Intention - Values that align with mine.

  - Integrity - Actions that align with their stated values.

  - Competence - Ability to execute to cause outcomes aligned with intentions.

  - Results - History of successful execution of aligned outcomes.

- The way to change things is to look at yourself and ask "what can *I* do?"

  - It's easy to point at others outside yourself as the locus of the blame.

  - It's hard to see, "what could *I* do differently?"

  - Our own actions are our primary locus of influence on the system.

- Suffering is pain and also the resistance to it.

  - Imagine a 2x2:

  - No pain + No resistance = No suffering

  - Pain + resistance = maximal suffering.

  - No pain + resistance = limited suffering.

  - Pain + no resistance = limited suffering.

- One thought-provoking way to interpret your dreams: assume that every character is actually you.