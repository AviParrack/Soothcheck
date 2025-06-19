# 9/3/24

*One day later than normal due to Labor Day holiday.*

- A concept I love: [<u>ofuda</u>](https://en.wikipedia.org/wiki/Ofuda).

  - It’s an ancient Shinto religious practice, but has taken on a new life in Anime.

  - At Shinto shrines, you write specific phrases on paper and attach them to an object to give them good luck.

  - In Anime, the phrases are spells, and attaching to them an object enchants it.

  - This is not too unlike programming!

  - If you have just the right incantation, you can imbue the computer with the ability to take action.

  - Ted Chiang also explored this idea in his story “Seventy-Two Letters”, where the inscriptions are put in the mouths of Golems.

  - Before, the magical spells of programming, an arcane and unforgiving magic system, were known to only a precious few that had mastered the art through onerous study.

  - But now LLMs can write the enchantments for you; you can simply write the enchantments in english!

- It used to be that programming “spells” could only create normal objects.

  - Normal software is like a wind-up toy, an automaton.

    - Just does precisely what it was told to do, no judgment.

    - Cheap to execute!

  - But now software can also use LLMs in its execution.

    - LLMs are like magic in software.

    - Software with an LLM inside is squishy, alive, emergent but also a bit unpredictable.

    - An LLM can use its judgment, which the user might not want.

    - A little bit of LLM magic enchants a bit of software. A lot gives it something that looks like its own agency.

  - LLMs can produce both plain old automaton software, or magical and alive software.

- LLMs scramble the cost equation of software.

  - Before, software was expensive to write, cheap to run.

  - But now LLMs make software much cheaper to write.

    - At least, for mini, disposable software.

  - And LLMs also are a new kind of ingredient in software that can make it magic.

    - But software that uses LLMs to execute is quite expensive to run!

  - Software used to be expensive to write, cheap to run.

    - Now it’s relatively cheaper to write, relatively more expensive to run.

    - This flips a number of foundational assumptions in our industry on their heads!

  - A pattern for software that is cheap to write and cheap to run:

    - Use an LLM to write and configure the software.

    - But ensure that the configured software doesn’t require any LLMs to execute.

    - You only need to use the LLM in proportion to how often the software needs to *change*, not how often it needs to *run*.

- It’s really easy to use Claude to write 250 line webapps.

  - If it does the thing it's supposed to, great.

  - If it doesn't, who cares, it's disposable and cheap.

  - Scoped software for 250 lines, you'll never have to write again.

  - Engineers used to have to care how it worked.

  - But for small things that were cheap to do, who cares.

  - Software isn't precious anymore. It's disposable. Who cares how it works?

  - LLMs are great at writing frontend code because you can easily poke at the UI of the thing and verify it does what it’s supposed to.

  - It’s harder to poke at if it's a backend thing to verify it works.

- Software written by LLMs is merely good, not great.

  - If it’s small and similar to existing software, it’s typically good enough.

  - But if it’s larger, or unlike existing software, it’s less likely to be good enough.

  - Imagine a system that pieces together novel software by assembling sub-components.

  - The sub-components could have been written by:

    - 1\) humans in the past (highest quality)

    - 2\) written by LLMs and approved/tweaked by humans (moderate quality)

    - 3\) hallucinated by LLMs on demand. (lowest quality)

  - If a sub-component of the third class is used to compose a larger bit of software that doesn’t work, a 1% kind of user who is comfortable programming could pop the hood and tweak it.

    - That tweaked version can now be put on the shelf for everyone else to benefit from in the future.

  - This creates a kind of self-ratcheting quality.

    - The improvements by any human in the past improves the quality of the software generated for every human in the future.

- Apps lead to aggregation.

  - What if you could disaggregate software?

  - What if you could create disposable software on demand?

  - You’d need new laws of physics for software.

- New laws of physics for software: contextual flow control.

  - This is a name I came up with for how to combine a few existing concepts in a new way to create a system with very different properties.

  - Contextual Integrity is Helen Nissebaum’s notion of the platonic ideal of privacy.

    - “My data is used in line with my interest and intent”

  - Information Flow Control is an applied math framework that’s been around for 50 or so years.

    - It allows making formal statements about the confidentiality and integrity of information as it flows through a graph of operations.

    - Facebook just shared that they [<u>use it as a fundamental concept underpinning their internal data privacy aware infrastructure</u>](https://engineering.fb.com/2024/08/27/security/privacy-aware-infrastructure-purpose-limitation-meta/).

  - Contextual flow control is an approach that applies Information flow control concepts in a specific way to software that has been contained within independent modules, so all data flow across modules can be tracked and verified, allowing contextual integrity for novel software.

  - Think of it like **type checking for privacy policies**.

  - If the software “compiles”, then it by construction does not have any data flows that go against a user’s privacy goals.

  - Engineers know the power of type checking: it’s more work up front, but gives you significantly more confidence that your software is correct.

    - This allows modifying your software in a tighter / faster iteration loop.

    - Make the change that you want to see, and then keep resolving compiler errors until the software compiles, and you’re done.

    - LLMs can do this iteration cycle for you automatically; they’re very good at taking the structured output of a compiler error and tweaking software to improve it.

- If you make a new type of software that has new laws of physics, you can only run it in places that you can verify the physics are upheld.

  - Imagine software that uses Contextual Flow Control.

  - If you run the software on your own machine, you can trust that it was not tampered with and the laws of physics were faithfully upheld.

  - But that goes out the window when you run the logic in the cloud.

  - That means that software with new laws of physics would be limited to only users who could successfully run software from the command line–a small fraction of the population.

  - But if you use [<u>Private Cloud Enclaves</u>](https://docs.google.com/document/u/2/d/1w1RbFtk2AB1QjrmPMr3BWcrhv6uJiYzhPLQd07DN2Bc/edit#heading=h.pipyx2w5mv99), a user can verify that the laws of physics are being faithfully executed… even in VMs in the cloud that they aren’t running themselves.

  - This could raise the ceiling to anyone who is comfortable using a website today.

- A new Operating System is incredibly hard to distribute.

  - The OS normally runs at the level of the hardware, and getting people to buy new hardware is a challenge.

  - That, and every new OS needs a whole new ecosystem of software; it’s very hard to get to a critical mass of software.

  - But what if you made an OS distributed in the browser?

  - As far as the browser is concerned, it's just any other webpage.

  - A pocket universe with alternate laws of physics that manifests in a browser tab.

- Why does the app ask you a question and then the OS asks you the same question?

  - E.g. the app asks if you want to turn on notifications, and then when you say yes the OS also asks if you want the app to turn on notifications.

  - Because the OS can’t trust any of the software in “userland” to not be trying to cheat.

  - The OS has to show UI it trusts, that it knows shows the user relevant information to allow them to make an informed choice.

  - What if you could make it so the userland UI could be trusted in some cases?

  - You’d remove all of the obnoxious permission prompts; your common sense, obvious actions in the app would be sufficient to let the OS know you consented.

- You have to trust all of the code inside the same container where the sensitive operation happens.

  - Typically the container is the process, the app, the origin: a chunky container with a whole lot of stuff inside.

    - This is a *lot* of code to have to trust!

  - But if you break up the code into a series of contained modules with limited ways to interact except through audited channels, then often the amount of code that has to be deeply trusted because it’s in the same container as the sensitive operation is quite small.

- By default, users don’t have sovereignty over their data in the cloud.

  - To regain sovereignty, the obvious step is to build local first software.

  - The users have sovereignty over their data, because the data remains on their turf.

  - Local gives you privacy, agency, control.

  - But the cloud gives you efficiency, scale, reliability, power.

  - Going local first introduces a number of new challenges:

    - How do you do decentralized sync?

      - Any device might be arbitrarily out of sync, for an arbitrarily long amount of time.

    - How do you shrink down LLM inference to be efficient enough to run on device?

    - How do you accomplish code that every user has to trust is running in an environment that none of them can tamper with?

  - These are hard challenges!

    - In the cloud, these are all easy; you can presume a small number of beefy devices that are available practically 24/7.

  - [<u>Private cloud enclaves</u>](https://docs.google.com/document/d/1w1RbFtk2AB1QjrmPMr3BWcrhv6uJiYzhPLQd07DN2Bc/edit#heading=h.pipyx2w5mv99) provide a different approach.

    - The power of the cloud, the sovereignty of local.

- "Users will never understand that security architecture well enough to trust it"

  - They don't have to!

  - They just have to know that their more knowledgeable friend trusts it.

  - This can go, inductively, all the way down to the small number of security professionals who read the white paper and even inspect the code themselves.

  - This knowledge can take time to diffuse through a population, which is why people erroneously believe that people won’t be able to trust a novel architecture.

- Imagine that you had changed the laws of physics of software.

  - Where would you go to find use cases that might make sense to try?

  - One source is the graveyard of failed startup apps.

  - Apps that created user value… but had no viable business.

  - But if the laws of physics have changed to make the overhead of distribution, or the amount of bespoke data necessary to make it viable have changed, they might be viable now!

- A use case I want, but that’s completely impossible in today’s laws of physics.

  - “You have \$5,000 in accumulated credit card points. In the past you’ve used your rewards points to splurge on beach resort vacations with your husband. Here are four options for high-rated beach resorts you’ve never been to that fit within your calendar’s availability and the ideal weather for each location.”

  - To do this would require having your calendar data, your financial data, your travel history, etc all in one place, with software that could have access to all of it to generate new insights.

  - In the app model that would be terrifying!

- For a use case to be viable today, the user has to give it the data it needs to operate.

  - Today the unit of computation is the app.

  - The user has to be comfortable giving the app the data it needs to do the use case.

  - That requires a viable business model to exist for the app to be viable.

    - That’s a very high bar!

  - Imagine a power-law curve of all conceivable use cases.

    - Only the ones that are extremely valuable, say the top 1%, are above the bar of being important enough to get over the “an app is worth it” hump.

    - But there’s a *huge* tail of plausible use cases that are not viable.

    - So much value that is not created!

  - Every new use case today has a cold start problem.

    - As a user, you have to hope that either:

    - 1\) An app you already use and has the data decides to add the incremental use case.

      - But as the app gets more bloated and complex, and the org that produces it gets more bloated and complex, it gets harder and harder for the org to add incremental features to the app.

    - 2\) a new app will do it

      - But the app has to clear a very high cold start problem to get users to get over the hump of installing it and giving it data, and has to have a viable business model embedded.

  - In today's laws of physics, apps can't go where the data is, because data is kept inside the castle of the other origin.

    - So everyone competes to be the place where the data is.

  - But imagine if there was an ecosystem, and within it, each use case could operate over precisely the data it needed with no friction, safely, without any ability to use it for other purposes.

    - The ecosystem itself would have a large cold start problem, like any new app.

    - But every use case within the ecosystem–from any third party participating in it–would have *significantly* less of a cold start problem.

    - As more and more activity happened in the ecosystem (more use cases supported, more users with more data onboarded), the ecosystem would get more and more momentum.

    - Each new use case added to the ecosystem would make the incremental value of the ecosystem stronger, giving more of a pull for incremental users to be willing to jump in.

    - Every use case that works helps decrease the cold start problem for every other use case.

    - All of the momentum of all of the use cases in the ecosystem aggregated into the momentum of the collective ecosystem.

    - It would be wildly unlike the current world.

    - This ecosystem would have a massive moat.

    - Competing apps or ecosystems would have to overcome the significant amount of data momentum of the ecosystem.

    - And if the ecosystem were open, then it would make more sense to just join in, instead of trying to fight it.

- It takes expertise to figure out how to bend the rules in a provocative and innovative way.

  - But it takes time to develop the knowhow and expertise in a given context.

    - The only way to develop it is to do it, a kind of apprenticeship.

  - LLMs help with the "good enough" versions of bending the rules, but you don't develop the knowhow that allows great versions.

  - Will LLMs take away apprenticeship and the ability to innovate in interesting ways?

- There are some cases where you want to not be average (areas where you have calibrated skills).

  - There are other cases where you just want to be good enough (because you have few skills in it).

  - If you're in a large company, there's a person that has the expertise of marketing.

    - So it’s easier to just ask the expert to do it.

  - If you're a solo entrepreneur, the LLM-assisted marketer is way better than what you’d do on your own.

  - A startup founder needs to do all of the tasks, but in a big company, you have experts for everything.

    - "Don't touch that! That’s for the marketing team to do!".

    - How could you learn?

  - With LLMs, everyone gets to be just kind of average (or slightly below average) at basically every knowledge work task.

  - That might make skills where you’re exceptional stand out more.

- AI might cause a regression in craftsmanship and understanding of the art of making things.

  - You're getting the result but without the learning, understanding, cost, meaning.

  - A focus on just the result, not the process.

  - But the learning comes from the process, not the result.

  - LLMs snapshotted the outcome of human craftsmanship to date, and made it easy to recreate the results.

  - But by making the process easy, we might have removed the ability to generate more craftsmanship!

- A metaphor for LLMs: an electric bicycle for the mind.

  - Bicycles are about extending human agency but you're very much still steering.

  - If you already know how to bike, you can do an electric bicycle, it has all the same affordances!

  - In tech we jump right to "self driving cars", and then it turns out to be significantly harder than we thought to get there.

  - Doing something like electric bikes for the mind is in the adjacent possible; unlocks value immediately and provides a base to grow.

- The things that will hurt you the most are the things you didn't realize you didn't know.

  - The [<u>four stages of competence mode</u>](https://en.wikipedia.org/wiki/Four_stages_of_competence)l:

    - Unconscious incompetence - Wrong intuition

    - Conscious incompetence - Wrong analysis

    - Conscious competence - Right analysis

    - Unconscious competence - Right intuition

  - We're all idiots, and we don't know it.

- A [<u>new kind of burn on Twitte</u>](https://x.com/a_4amin/status/1829491695408627724)r: “your product is so simple I could build it as an Artifact in a few minutes”

- An [<u>insightful tweet</u>](https://x.com/emollick/status/1800289813301706862?s=46&t=vBNSE90PNe9EWyCn-1hcLQ) from Ethan Mollick:

  - “It wasn’t the steam engine alone that caused the Industrial Revolution. It was the thousands of specific machines invented by skilled craftsmen that used steam to augment or do existing work that created the Revolution

  - Assuming AGI doesn’t happen, AI will evolve in similar ways”

- Progress comes from disconfirming evidence.

  - And yet it is emotionally hard to receive disconfirming evidence.

  - It makes people do a defensive crouch--which then makes it even harder for them to receive it, in a toxic spiral.

  - How to create an environment where it can be low stakes enough so people receive it without getting defensive?

  - That requires careful work to create and maintain a psychologically safe space.

  - Psychologically safe spaces create the space to receive disconfirming evidence; to have unsafe thinking; for maximal growth and innovation.

- At the frontier of innovation, there aren’t words for what you’re doing yet.

  - Kevin Kelly: “Try to work on an area where there’s no words for what it is that you do… When you are ahead of language, that means you are in a spot where it is more likely you are working on things that only you can do. It also means you won’t have much competition.”

  - [<u>David Whyte</u>](https://anotherjesse.com/posts/being-misunderstood/): if you're being fully understood about the things you're doing, then they aren't that interesting.

- It’s easier to be a trim tab when you interact with people on a regular cadence.

  - A trim tab is a small adjustment surface on a wing that with a small movement adjusts the flow.

  - Buckminster Fuller: [<u>“Call me trim tab”</u>](https://www.themarginalian.org/2015/08/21/buckminster-fuller-trim-tab/)

  - If you only meet with someone sporadically or on demand, then it might be too late for a small adjustment to lead to a better result.

    - They might not have realized they’d made a decision that put them on a path that wasn’t ideal.

    - You’ll need to apply significant effort to adjust their direction… effort they might resent or resist.

  - But if you meet with them regularly, you’ll notice things that could use some adjustment very early, when only the teensiest, lightest touch adjustment can put them on a much better path… perhaps without them even realizing you’re nudging them at all.

  - This is why recurring 1:1s are great.

  - Smooth, laminar flow, the lightest touch of adjustments.

- [<u>Someone on Twitter notes</u>](https://x.com/cjhandmer/status/1829221690074968376?s=46&t=vzxMKR4cS0gSwwdp_gsNCA) that, unlike software, VC hasn’t created a repeatable playbook for hardware.

  - When you have a repeatable playbook that’s how you know you’re in a rut that could be dangerous if the conditions change.

  - Efficiency is in tension with resilience.

- We normally try to avoid challenge, because it's uncomfortable.

  - Couples counseling (or any coaching or therapy) forces you to have challenge on a regular basis.

  - Normally the world is bombarding you with challenge, so you can't help but learn.

  - But when you're extremely powerful, more relationships are sycophantic to you; they don't challenge you.

  - You get less and less challenge, so you get less and less emotional and intellectual growth.

- Converging too quickly is dangerous in an ambiguous situation.

  - If you have to converge immediately, you pick whatever random idea you happen to believe in that moment and lock it in.

  - It gives you more stability, but in a way that makes you feel less stressed but actually puts yourself in a more brittle / dangerous position in an ambiguous or shifting environment.

  - There's value in the superposition, absorbing new information, experimenting with different directions, getting better calibrated bets before committing to one.

  - The superposition will be stressful; people who are detail oriented will push for premature convergence.

- If you don’t know what the right answer is and you have a limited runway, optimize for building something adaptable and reaching a level of good enough as cheaply as possible.

- When you’re stumbling through the fog, it’s easier to give up if you don’t know if there’s anything worth getting to on the other side.

- The ripples on the surface are what you can see (and thus what everyone pays attention to and talks about all the time), but what actually matters is the current underneath.

- Lower pace layers are often very hard to change.

  - But every so often an opportunity pops up when things have been thrown into chaos by a new disruptive force that scrambles cost structures, and invalidates the accumulated knowhow of the existing system.

  - In those rare situations, there are sometimes valid changes in lower pace layers.

  - If you can successfully change a lower pace layer, *tons* of things can change.

    - You get almost unbelievable leverage.

  - The trick is being in such a chaotic situation, and then identifying the seed crystal of a new approach in that swirling situation.

- Every time you tell a story it becomes more of a caricature.

  - You keep the details that fit into or accentuate the narrative, and de-emphasize the parts that don't.

  - You keep the details that are interesting or intriguing or distinctive or funny, and factor out the parts that don't.

  - The same is roughly true when you retrieve a memory.

    - The memory is not stored as a whole snapshot; it was stored as the interesting diff from the background expectations, at the time the memory was stored.

    - It's like the dinosaur DNA in Jurassic Park.

      - There are gaps in the DNA, so you need something to fill the gaps.

      - You use a new baseline, like frog DNA, to fill in the missing parts.

    - When you retrieve the memory, your baseline understanding may have evolved.

    - So you fill in the gaps in the memory with different things, so the memory is different.

    - Things that compress well, that fit a coherent narrative, are easier to hold on to.

    - Every time you retrieve the memory you change it and then restore it.

    - The original memory fades away, replaced by its caricature.

- A general purpose internal politics move to navigate a thing that people could freak out about: bring it up in 1:1 before sharing it in a group meeting.

  - When people are surprised by negative (or even ambiguous) information in front of other people, there’s a chance they’ll freak out, they feel out of control.

  - If you brief everyone before the group meeting individually, everyone will have enough information to not freak out.

  - People know that they were briefed behind closed doors.

    - They might even feel special that they were in the loop before the official announcement.

    - But because everyone else was briefed behind closed doors, they don’t see each other’s briefing.

    - It can feel like they’re special, even if everyone got a briefing.

  - Briefing everyone before is easy if you have recurring 1:1s.

    - Just put an agenda item for each 1:1.

    - Once everyone is briefed, you can bring it up at the team meeting.

    - This means that you can bring it up as quickly as the longest 1:1 recurrence schedule.

    - If you don’t have a 1:1 with one of the members you have to hope they will be OK with it (a bit of a leap of faith), or will see that a critical mass of others are OK with it and be OK, too.

- "Just trust us, we know what's good for you."

  - Don't place your trust in an entity that has demonstrated it only thinks about the first-order effects of their actions.

  - Trust is about the track record of indirect effects, the holistic, non-transactional effects.

  - Users shouldn’t trust an industry that wants all the power but with a complete abdication of the responsibility to think through the implications.

- A tree looks beautiful as an outcome, but it's engaged in a brutal, constant struggle for its existence.

  - The generative process of competition in an ecosystem is fundamentally something that causes death.

  - If the context weren't brutal to survive in, then there would be no impetus to change.

- The swarm evolves even if none of the particles that make up the swarm change.

  - The aggregate of the swarm is the summation of the velocities of the particles.

  - The swarm can change when the particles change… or when the set of particles included in the population changes.

    - For example, new particles joining the swarm.

    - Or existing particles dying or leaving the swarm.

  - As long as the swarm subsets particles (that is, the ones it keeps) with a consistent bias, the aggregate vector of the swarm will become coherent and strong.

  - You don’t need active selection by some judgment; if the unfit particles are selected out by failing to survive then the swarm gets the velocity of the fittest in this environment, and the selection bias is emergent but also with a consistent bias.

  - The adaptation works as long as there's a consistent bias in the selection.

  - If the particles can change (e.g. the beliefs, skills, incentives of humans in an org) then the swarm can change even without particles dying.

  - But if the particles can’t change (e.g. an organism's genome) then the only way for the swarm to adapt is for particles to continuously be dying with some bias.

- Someone who has an OODA loop much faster than the rest of the group will go off on their own, not bring people along.

  - But if they did bring people along they'd be on a short leash, way less able to scout out ahead.

  - How do you let them run ahead, try out different paths, and scout ahead and then come back to converge to point the way?

  - How do you collaborate with them as part of a coherent whole?

  - The [<u>Hash House Harriers</u>](https://en.wikipedia.org/wiki/Hash_House_Harriers) might provide a parallel.

    - These are non-competitive social running groups, where a trail is set and everyone runs it at the same time together.

    - The trail is marked by periodic markings of flour.

    - Every so often the trail ends in a “check”: an X of flour.

    - The trail then goes cold, and will pick back up somewhere maybe hundreds of feet away.

    - At a check, the runners have to fan out (a swarm!) to find where the trail picks back up.

    - When anyone finds it, they yell “on, on!” so that everyone else knows they found it and to follow them.

    - This has a nice dynamic: it allows slower runners to keep up.

    - The fastest runners get to the check first, and fan out in different directions.

    - Any individual fast runner likely picked the wrong direction to search for the trail, and will have to double back once it’s found.

    - The slowest runners will arrive at the check later, and will fan out only a little bit before the trail is found, and won’t have to double back at all.

    - This clever little collective intelligence mechanism is one reason Hash House Harriers are such popular and enduring social groups.

- How long of a leash should you give someone to prototype in a divergent way?

  - The answer is related to how likely they are to find something great… and how likely they are to successfully do the deferred convergence work if they find something great.

  - How likely they are to do both of these things is based on their teammate's priors of interacting with that team mate in the past.

- An open system is one no one can tell to "stop".

  - Because as long as some participants invest in it, it will keep growing and adapting.

  - If you sit it out, you’ll get left behind.

  - This is not always a good thing! Sometimes it forces society to keep pressing forward on a technology that everyone is nervous about the implications of, but that gives some “edge” to those who deploy it.

- Hyper centralization looks superficially similar to a healthy ecosystem.

  - But it's the kayfabe of competition.

  - An ecosystem where the key aliveness that makes it antifragile is dead, leaving an efficient, brittle husk.

  - The efficiency gains are real and significant, but they come at the cost of the indirect health of the system (its ability to innovate and adapt) being reduced.

  - Let’s dive into one specific notional example.

    - Independent coffee shops used to be a third place.

      - A place that isn’t home or work for members of the community to congregate and interact.

    - The owner of each shop, since they are an owner, is more likely to have an “owner mindset”, to create a place to hang out that was an anchor for the community.

      - The owner mindset is more likely to invest in things with indirect benefits.

    - A franchise owner is more like a renter mindset; can't change much if corporate doesn't let them.

    - But now imagine every coffee shop is owned by one company.

    - The system is now in a supercritical state, right on the edge of an instantaneous state change.

    - Now, some MBA at that company goes "people hanging out here isn't helping the bottom line" so proposes having a policy of "buy a coffee or GET OUT".

    - This is rolled out nationwide instantly, and now collapses all third places at once, systemically.

    - Whereas when they were independent, many coffee shop owners might have made different decisions, and we wouldn't lose all of the third places at once.

    - But now a single boneheaded decision could destroy all third places in the country in a snap.

  - This is just one example, but this kind of centralization has happened in nearly every industry in the past few decades.

- A boring challenge is the worst of all worlds.

  - A slog.

  - How can you find *intriguing* challenges?

  - An intriguing challenge is one that gives you the joy of discovery and growth as you incrementally move through it, helping you keep motivation at each step.

- A general pattern for creating value: take a thing that robustly works today but is manual and specific, and systematize it to make it automatic and more general.

  - You make a lot of one-off additional investments that previously weren't viable (worth the bespoke effort) now viable, automatically, because they can use the now generalized foundation.

  - You also reduce the carrying cost of special cases, freeing up more resources to be spent on innovation.

  - I’ve called this Primitive Archeology in the past.

  - If you have an existing, successful system, this is a no-brainer strategy that will almost certainly unlock value.